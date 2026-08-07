import h5py
import random
import re
import torch
from utils.data.transforms import (
    DataTransform, MRIAugmentor, AugmentConfig, equispaced_mask, canonical_offset,
)
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.utils.data._utils.collate import default_collate
from pathlib import Path
import numpy as np


def _worker_init_fn(worker_id):
    """멀티프로세스 dataloader에서도 결정적이 되도록 worker별 seed 파생 (재현성)."""
    base = torch.initial_seed() % (2 ** 31)
    seed = (base + worker_id) % (2 ** 31)
    np.random.seed(seed)
    random.seed(seed)


def _bbox_collate(batch):
    """return_bbox=True일 때 가변 길이 bbox를 배치로 묶기 위한 collate.
    bbox는 항상 튜플 맨 끝이라(DataTransform docstring 순서: ...→bbox), 마지막 원소만 리스트로 유지하고
    나머지(mask/kspace/target/maximum/fname/slice[/scale][/grappa])는 기본 collate.
    → scale·grappa가 중간에 끼어도 인덱스 하드코딩 없이 동작(맨 끝=bbox 규약)."""
    core = default_collate([b[:-1] for b in batch])
    bboxes = [b[-1] for b in batch]
    return (*core, bboxes)


def _acc_key(name):
    """파일명에서 'acc4_1', 'acc8_1' 같은 (가속률_인덱스) 토큰을 뽑아 페어링 키로 사용.
    이미지(img_...)와 k-space(kspace...) 파일명이 달라도 같은 샘플끼리 짝지을 수 있게 한다."""
    m = re.search(r'(acc\d+_\d+)', name)
    return m.group(1) if m else name


KFOLD_ROOT_SEP = "__"


def volume_id(root, name):
    """여러 root를 한 데이터셋으로 합칠 때 쓰는 **고유** volume 식별자.

    train/과 val/에는 이름이 같은 **서로 다른** 볼륨이 30쌍 있다
    (예: train/…/knee_acc4_1.h5 = 23슬라이스, val/…/knee_acc4_1.h5 = 31슬라이스).
    그래서 파일명만으로 볼륨을 식별하면 (a) fold 분할이 두 볼륨을 하나로 세고
    (b) validate()의 reconstructions[fname] 딕셔너리에서 두 볼륨의 슬라이스가 섞인다.
    root 이름을 앞에 붙여 그 충돌을 없앤다. 파일명으로도 쓰이므로(save_reconstructions,
    fold 채점 디렉터리) 경로 구분자가 아닌 '__'로 잇는다.
    root가 하나뿐이면 파일명을 그대로 쓴다 → 기존(비-K-Fold) 경로는 100% 동일하게 동작.
    """
    return f"{Path(root).name}{KFOLD_ROOT_SEP}{name}"


def _file_acc(name):
    """파일명에서 가속률 숫자('4'/'8' 등)를 뽑는다(MoE acc 필터용). 없으면 None."""
    m = re.search(r'acc(\d+)', name)
    return m.group(1) if m else None


def _norm_acc(acc):
    """요청 acc(args.acc: 'all'|'4'|'8'|4|8) → None(전체) 또는 '4'/'8'."""
    if acc in (None, 'all', ''):
        return None
    return str(acc)


class SliceData(Dataset):
    def __init__(self, root, transform, input_key, target_key, forward=False, volumes=None, acc=None,
                 num_adj_slices=1):
        self.transform = transform
        self.input_key = input_key
        self.target_key = target_key
        self.forward = forward
        # 2.5D 인접 슬라이스: 1이면 기존과 완전히 동일(추가 I/O 0). 3이면 s-1,s,s+1을 읽어
        # **코일축에 슬라이스-major**로 쌓는다 — (adj*C, H, W), [s-1의 C코일, s의 C코일, s+1의 C코일].
        # 이 배열이 promptmr_plus의 sens_reduce/sens_expand·감도 RSS 정규화가 전제하는 규약이다.
        self.num_adj_slices = int(num_adj_slices)
        if self.num_adj_slices % 2 != 1:
            raise ValueError(f"num_adj_slices는 홀수여야 한다(중앙 슬라이스가 있어야 함): {num_adj_slices}")
        # (image_path|None, kspace_path, slice_idx, vol_id) 튜플. image와 kspace를 **같은 root 안에서**
        # 짝지어 한 항목에 담는다. 예전엔 두 리스트에 따로 쌓고 같은 인덱스로 꺼낸 뒤 .name이 같은지로
        # 검증했는데, root가 둘이 되면 그 검증이 무력해진다(train/…/x.h5 와 val/…/x.h5 는 .name이 같다).
        # 페어를 만들 때 묶어두면 어긋날 여지 자체가 없다.
        self.examples = []

        roots = root if isinstance(root, (list, tuple)) else [root]
        keep = set(volumes) if volumes is not None else None
        multi = len(roots) > 1
        acc_want = _norm_acc(acc)   # MoE expert 학습용 가속률 필터('4'/'8'). None이면 전체.

        for r in roots:
            r = Path(r)
            for kspace_fname in sorted((r / "kspace").glob("*.h5")):
                if acc_want is not None and _file_acc(kspace_fname.name) != acc_want:
                    continue
                vol_id = volume_id(r, kspace_fname.name) if multi else kspace_fname.name
                if keep is not None and vol_id not in keep:
                    continue
                image_fname = None
                if not forward:
                    image_fname = r / "image" / kspace_fname.name
                    if not image_fname.exists():
                        raise ValueError(
                            f"k-space 파일 {kspace_fname} 에 대응하는 이미지 파일이 없습니다: {image_fname}"
                        )
                num_slices = self._get_metadata(kspace_fname)
                self.examples += [
                    (image_fname, kspace_fname, slice_ind, vol_id)
                    for slice_ind in range(num_slices)
                ]

        # ------------------------------------------------------------------
        # [임시·주석] utils/data 폴더에 평평하게 놓인 단일 샘플 파일 직접 로딩.
        #   데이터셋 미연결(로컬 테스트) 환경 전용. 원본 데이터셋 연결 시 위 [원본] 블록 사용.
        #   이미지(타깃) : img_knee_acc4_1.h5, img_knee_acc8_1.h5
        #   k-space(입력): kspace_knee_acc4_1.h5, kspace_knee_acc8_1.h5
        # acc 토큰(acc4_1 / acc8_1)으로 이미지·k-space를 짝짓는다(파일명이 달라도 페어링).
        # 인자로 받은 root(data_path) 대신 이 파일이 있는 utils/data 폴더를 사용한다.
        # ⚠️ 이 경로를 되살릴 땐 _acc_key로 페어를 만들어 self.examples에
        #    (image_path, kspace_path, slice_idx, vol_id) 형태로 넣도록 함께 고쳐야 한다.
        # ------------------------------------------------------------------
        # root = Path(__file__).resolve().parent
        # kspace_files = sorted(root.glob("kspace*.h5"))
        #
        # image_map = {}
        # if not forward:
        #     image_files = sorted(root.glob("img*.h5"))
        #     image_map = {_acc_key(f.name): f for f in image_files}
        #
        # for kspace_fname in kspace_files:
        #     num_slices = self._get_metadata(kspace_fname)
        #     self.kspace_examples += [
        #         (kspace_fname, slice_ind) for slice_ind in range(num_slices)
        #     ]
        #     if not forward:
        #         key = _acc_key(kspace_fname.name)
        #         if key not in image_map:
        #             raise ValueError(
        #                 f"k-space 파일 {kspace_fname.name} 에 대응하는 이미지 파일을 "
        #                 f"data 폴더에서 찾지 못했습니다 (키: {key})."
        #             )
        #         image_fname = image_map[key]
        #         self.image_examples += [
        #             (image_fname, slice_ind) for slice_ind in range(num_slices)
        #         ]


    def _get_metadata(self, fname):
        with h5py.File(fname, "r") as hf:
            if self.input_key in hf.keys():
                num_slices = hf[self.input_key].shape[0]
            elif self.target_key in hf.keys():
                num_slices = hf[self.target_key].shape[0]
        return num_slices

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        image_fname, kspace_fname, dataslice, vol_id = self.examples[i]

        with h5py.File(kspace_fname, "r") as hf:
            ds = hf[self.input_key]
            if self.num_adj_slices == 1:
                input = ds[dataslice]
            else:
                # 볼륨 경계는 clamp(자기 복제) — 첫/마지막 슬라이스에서 인접이 없으면 자신을 반복한다.
                # 0-padding 대신 clamp인 이유: 빈 코일 그룹은 감도맵 RSS 정규화에서 0 나눗셈이 된다.
                n = ds.shape[0]
                half = self.num_adj_slices // 2
                idxs = [min(max(dataslice + d, 0), n - 1) for d in range(-half, half + 1)]
                input = np.concatenate([ds[j] for j in idxs], axis=0)   # (adj*C, H, W) 슬라이스-major
            mask =  np.array(hf["mask"])
        grappa = None
        if self.forward:
            target = -1
            attrs = -1
            # forward(추론/leaderboard) 경로는 kspace 폴더만 읽음 → GRAPPA 미제공.
            # GRAPPA를 추론에도 쓰려면 leaderboard의 image_grappa 경로 확인 후 배선 필요(미확정, 나중/동료).
        else:
            with h5py.File(image_fname, "r") as hf:
                target = hf[self.target_key][dataslice]
                attrs = dict(hf.attrs)
                # GRAPPA 영상 prior — transform이 요청할 때만 로드(기본 off면 I/O도 0 → 기존과 동일).
                if getattr(self.transform, "return_grappa", False) and DataTransform.GRAPPA_KEY in hf:
                    grappa = hf[DataTransform.GRAPPA_KEY][dataslice]

        return self.transform(mask, input, target, attrs, vol_id, dataslice, grappa)


def verify_remaskable(dataset):
    """--remask 로더를 만들 때 1회. 재생성이 성립하는 두 전제를 데이터에서 직접 확인한다.

      (1) 생성기 동일성 — 딸려온 마스크가 equispaced_mask(W, 파일acc, o) 로 **어떤 offset o 에서든**
          비트 재현되는가. 재현되면 우리 생성기와 주최측 생성기가 같은 물건이라는 뜻이고,
          그게 "우리가 만든 마스크도 채점에 나올 법한 마스크"라는 이 기능의 근거다.
          (o 가 canonical=(W//2)%acc 와 다른 볼륨은 오류가 아니라 정보로만 센다 — 학습풀 200개는
           전부 canonical 이었지만 leaderboard 는 무작위라, 규칙이 아니라 관측일 뿐이다.)
      (2) full k-space — 마스크가 0인 열에 에너지가 남아 있는가(=마스크 곱 이전 저장).
          이미 언더샘플된 파일(leaderboard 형식)을 학습에 물리면 재생성이 "없는 라인을 켜는"
          꼴이 되어 입력이 조용히 망가진다.

    어느 하나라도 깨지면 10시간짜리 학습을 시작하기 전에 죽인다. 볼륨당 mask(W,) + 코일 1개
    슬라이스 1장만 읽으므로 160볼륨에 수 초다(슬라이스마다가 아니라 로더 구성 시 1회).
    ⚠️ PE 양끝은 full k-space 에서도 값이 0에 가까워, 에너지 검사는 중앙 절반으로 한정한다."""
    kfiles, seen = [], set()
    for _, kspace_fname, _, _ in dataset.examples:
        if kspace_fname not in seen:
            seen.add(kspace_fname)
            kfiles.append(kspace_fname)

    bad_gen, bad_full, noncanon = [], [], 0
    for kf in kfiles:
        acc = _file_acc(Path(kf).name)
        if acc is None:
            bad_gen.append(f"{Path(kf).name}: 파일명에 accN 토큰이 없어 가속률을 못 정함")
            continue
        acc = int(acc)
        with h5py.File(kf, "r") as hf:
            mask = np.asarray(hf["mask"])
            W = mask.shape[0]
            S = hf["kspace"].shape[0]
            line = np.asarray(hf["kspace"][S // 2, 0])       # 코일 1개·슬라이스 1장 (≈1.9MB)
        m = (mask >= 1)
        fit = next((o for o in range(acc)
                    if np.array_equal(equispaced_mask(W, acc, o) >= 1, m)), None)
        if fit is None:
            bad_gen.append(f"{Path(kf).name}: W={W} acc={acc} — 어떤 offset 으로도 재현 불가")
        elif fit != canonical_offset(W, acc):
            noncanon += 1
        lo, hi = W // 4, W - W // 4
        unsampled = ~m[lo:hi]
        if unsampled.any() and float(np.abs(line[:, lo:hi][:, unsampled]).max()) <= 0.0:
            bad_full.append(f"{Path(kf).name}: 마스크 밖 열 에너지 0 — 이미 언더샘플된 파일")

    if bad_gen or bad_full:
        msg = ["--remask 전제 검증 실패 — 재생성한 입력이 물리적으로 틀린 물건이 된다."]
        if bad_gen:
            msg += [f"  [생성기 재현 불가 {len(bad_gen)}건]"] + [f"    {s}" for s in bad_gen[:10]]
        if bad_full:
            msg += [f"  [full k-space 아님 {len(bad_full)}건]"] + [f"    {s}" for s in bad_full[:10]]
        raise ValueError("\n".join(msg))

    print(f"[remask] 전제 검증 통과 — {len(kfiles)} volume "
          f"(생성기 비트 재현 · full k-space" + (f" · non-canonical offset {noncanon}개)" if noncanon else ")"))


class HoldoutTailSampler(Sampler):
    """hold-out 전량 + train 풀에서 (n − hold-out)개 무작위 추출 → 둘을 섞어 n개.

    epoch 마다 다시 뽑히는 근거: train_part.train() 이 epoch 마다
    train_loader.generator.manual_seed(seed + epoch) 를 부르고, 이 샘플러가 그 generator 를
    **공유**한다. set_epoch 같은 별도 배선이 없는 이유다(따로 만든 generator 를 주면 재추첨이
    조용히 사라져 두 epoch 이 같은 부분집합을 본다).
    n ≤ hold-out 개수인 퇴화 상황에서는 hold-out 에서 n 개만 낸다 — __len__ 과 실제 개수가
    어긋나면 스케줄러 horizon 이 거짓말을 한다.
    """

    def __init__(self, hold, pool, n, generator):
        self.hold, self.pool, self.n, self.generator = hold, pool, n, generator

    def __len__(self):
        return self.n

    def __iter__(self):
        k = max(0, self.n - len(self.hold))
        drawn = torch.randperm(len(self.pool), generator=self.generator)[:k].tolist()
        idx = self.hold + [self.pool[i] for i in drawn]
        order = torch.randperm(len(idx), generator=self.generator).tolist()
        return iter([idx[i] for i in order][:self.n])


def create_data_loaders(data_path, args, shuffle=False, isforward=False, is_train=None, volumes=None,
                        acc_override=None, holdout=None):
    if isforward == False:
        max_key_ = args.max_key
        target_key_ = args.target_key
    else:
        max_key_ = -1
        target_key_ = -1

    # train/val 구분: 명시 안 하면 shuffle=True를 train으로 간주(train_part.py는 train만 shuffle=True).
    # 증강은 반드시 train에만 적용, val/forward는 원본 그대로(검증은 진짜 실력 측정).
    if is_train is None:
        is_train = bool(shuffle) and (not isforward)

    # ------------------------------------------------------------------
    # augmentation 설정 — train.py args에서 읽음(없으면 getattr 기본값 = off).
    # 남은 증강은 좌우 flip 하나뿐이다(mask thinning·위상·노이즈는 2026-07-28 제거).
    # --aug-on 기본값이 off라, 안 넘기면 기존 학습과 100% 동일하게 동작한다.
    # (인자 정의는 train.py, 매 epoch set_epoch 배선은 train_part.py — 둘 다 완료.)
    # ------------------------------------------------------------------
    aug_on = getattr(args, "aug_on", False)
    cfg = AugmentConfig(
        on=aug_on,
        schedule=getattr(args, "aug_schedule", "exp"),
        delay=getattr(args, "aug_delay", 0),
        total_epochs=int(getattr(args, "num_epochs", 1)),   # T = 런 길이 → 스케줄이 자동 스케일
        exp_decay=getattr(args, "aug_exp_decay", 5.0),
        max_prob=getattr(args, "aug_max_prob", 0.55),
        affine=getattr(args, "aug_affine", True),
        rot_max=getattr(args, "aug_rot_max", 5.0),
        scale_max=getattr(args, "aug_scale_max", 1.1),
        seed=getattr(args, "seed", 430),
    )
    augmentor = MRIAugmentor(cfg) if (is_train and aug_on) else None
    # C1: bbox는 **학습 전용** 신호(bbox 가중 손실). val/leaderboard 로더에는 절대 붙이지 않는다 —
    #     검증은 자연 분포·기존 언패킹을 그대로 유지해야 지표가 거짓말을 안 한다.
    return_bbox = getattr(args, "return_bbox", False) and is_train

    # ---- 확정 전처리 플래그 (기본 전부 off = 기존 학습과 100% 동일) ----
    #   켤 때는 network/학습 코드(train_part.py 등)의 배치 언패킹도 함께 확장해야 함
    #   (반환 튜플 형식은 DataTransform docstring 참고). off면 6-튜플 그대로.
    normalize = getattr(args, "normalize", False)
    readout_crop = getattr(args, "readout_crop", False)
    crop_h = getattr(args, "crop_h", 384)
    norm_percentile = getattr(args, "norm_percentile", 99.0)
    return_grappa = getattr(args, "return_grappa", False)
    # 효율 레버(후보): SVD coil compression. 기본 off = 기존과 동일. 배치 SME와 택1.
    #   켜면 kspace coil축이 n_virtual_coils로 축소되어 소비부(train_part 등)에 전달됨.
    coil_compress = getattr(args, "coil_compress", False)
    # 2.5D 인접 슬라이스 — 코일축에 슬라이스를 쌓으므로 코일축을 섞는 기법과 배타적이다.
    num_adj_slices = int(getattr(args, "num_adj_slices", 1))
    if num_adj_slices > 1 and coil_compress:
        raise ValueError(
            "--num-adj-slices > 1 과 --coil-compress 는 함께 쓸 수 없다. coil compression은 코일축을 "
            "SVD로 섞는데, 인접 슬라이스가 그 축에 쌓여 있어 서로 다른 슬라이스가 한 가상코일로 뭉개진다.")
    n_virtual_coils = getattr(args, "n_virtual_coils", 8)
    # B5: 손실·검증을 채점 정합 foreground 영역으로 제한(기본 off). train/val 양쪽 로더에 부착
    #     (forward는 target 없어 DataTransform이 내부적으로 미포함) → 소비부 train_epoch/validate 확장.
    fg_loss = getattr(args, "fg_loss", False)

    # ---- 마스크 재생성 (기본 off = 기존과 비트 동일) ----
    #   --acc 는 "어떤 **파일**을 읽을지", --remask 는 "그 k-space 에 어떤 **마스크**를 씌울지"라
    #   축이 직교한다. 그래서 ft 는 --acc all --remask 4|8 로 준다: 볼륨 풀은 160개 전부,
    #   마스크는 그 전문가의 가속률. 필터를 좁히면 절반만 학습하므로 오조합을 막는다.
    #   val/forward 로더에는 절대 안 붙인다 — 검증은 딸려온 마스크(=자연 분포)로 해야
    #   과거 런과 비교되고 채점 분포와도 맞는다.
    remask = str(getattr(args, "remask", "off"))
    remask_acc = None if remask in ("off", "") else int(remask)
    if remask_acc is not None and not is_train and not isforward and acc_override is None:
        # **val 로더만** 그 전문가의 acc 볼륨으로 되돌린다(재생성은 안 한다).
        #   호출부가 안 정해줬으면 여기서 --remask 로부터 파생한다 — 고정분할 경로(비-K-Fold)에서
        #   빠뜨리면 acc4 전문가가 acc8 섞인 val 로 best_model 을 고르게 된다.
        #   forward(추론)는 제외한다: 채점은 주어진 파일을 전부 복원해야 하므로 여기서 걸러버리면
        #   볼륨이 통째로 빠진다. 지금은 reconstruct.py 에 --remask 플래그가 없어 도달 불가지만,
        #   나중에 체크포인트 args 를 그대로 물려주는 경로가 생기면 조용히 터질 자리다.
        acc_override = remask
    if remask_acc is not None and is_train:
        if _norm_acc(getattr(args, "acc", "all")) is not None:
            raise ValueError(
                f"--remask {remask} 는 --acc all 과만 쓴다 (지금 --acc={getattr(args, 'acc', 'all')}). "
                "필터를 좁히면 볼륨 풀이 절반으로 줄어 이 기능의 목적(80→160 volume)이 사라진다.")
        if return_grappa:
            raise ValueError(
                "--remask 와 GRAPPA 는 배타적이다. image_grappa/image_input 은 **딸려온 마스크** "
                "기준 산출물이라, 재생성한 입력과 짝지으면 제거한 라인의 정보가 옆문으로 새어 "
                "실제 추론엔 없는 지름길을 학습한다.")

    # ---- 마스크 증강 (--mask-aug, 기본 off) ----
    #   --remask 와 같은 재생성 경로를 쓰되 --acc 제약은 없다 — offset 모드는 가속률을 안 바꾸므로
    #   볼륨 풀을 넓힐 필요가 없다. train 로더 전용인 것은 동일(val/forward는 딸려온 마스크 그대로).
    mask_aug = str(getattr(args, "mask_aug", "off"))
    if mask_aug != "off" and is_train and return_grappa:
        raise ValueError(
            "--mask-aug 와 GRAPPA 는 배타적이다(--remask 와 같은 이유). image_grappa 는 딸려온 "
            "마스크 기준 산출물이라 재생성한 입력과 짝지으면 제거한 라인 정보가 옆문으로 샌다.")

    data_storage = SliceData(
        root=data_path,
        transform=DataTransform(
            isforward, max_key_,
            augmentor=augmentor, is_train=is_train, return_bbox=return_bbox,
            normalize=normalize, readout_crop=readout_crop, crop_h=crop_h,
            norm_percentile=norm_percentile, return_grappa=return_grappa,
            coil_compress=coil_compress, n_virtual_coils=n_virtual_coils,
            return_fg_mask=fg_loss,
            remask_acc=(remask_acc if is_train else None),
            remask_offset=getattr(args, "remask_offset", "uniform"),
            remask_seed=getattr(args, "seed", 430),
            mask_aug=(mask_aug if is_train else "off"),
        ),
        input_key=args.input_key,
        target_key=target_key_,
        forward=isforward,
        volumes=volumes,
        acc=(acc_override if acc_override is not None else getattr(args, "acc", "all")),
        num_adj_slices=num_adj_slices,
    )

    if (remask_acc is not None or mask_aug != "off") and is_train:
        verify_remaskable(data_storage)   # 재생성의 전제(생성기 동일 · full k-space)는 mask_aug도 똑같이 필요
        print(f"[remask] train 로더 마스크 재생성 — remask={remask} · mask-aug={mask_aug} · "
              f"offset={'재추첨' if mask_aug != 'off' else getattr(args, 'remask_offset', 'uniform')}")
    elif remask_acc is not None:
        print(f"[remask] val/forward 로더: 재생성 안 함 (딸려온 마스크 그대로, acc 필터={acc_override})")

    # 재현성: 증강/결정적 로더를 쓸 때 generator·worker seed 부착.
    # shuffle=True(train)에도 항상 부착한다 — iter 단위 재개가 "epoch E의 순열"을 재현할 수 있어야
    # 저장된 iter까지 정확히 건너뛸 수 있다. generator가 None이면 RandomSampler가 매 epoch 전역
    # RNG에서 seed를 새로 뽑아 순열이 재현 불가가 된다. train_part.train()이 epoch마다
    # manual_seed(seed + epoch)로 다시 심는다.
    num_workers = getattr(args, "num_workers", 0)
    use_det = aug_on or getattr(args, "deterministic_loader", False) or shuffle
    seed = getattr(args, "seed", None)
    generator = None
    if use_det and (seed is not None):
        generator = torch.Generator()
        generator.manual_seed(int(seed))

    # --holdout-tail: hold-out(val fold) 볼륨까지 학습에 넣되 epoch 당 슬라이스 수는 본 런과 같게 둔다.
    #   n 은 --max-iters 에서 **유도**한다(그 수를 절단 장치로 쓰는 게 아니라 숫자만 빌린다) —
    #   len(loader) ≤ max_iters 라 train_epoch 의 break 가 안 걸리고, 그래서 epoch 전체에 흩어진
    #   hold-out 이 무작위로 잘려나갈 일이 없다. generator 는 DataLoader 와 **같은 객체**를 준다.
    sampler = None
    if holdout:
        hold = [i for i, e in enumerate(data_storage.examples) if e[3] in holdout]
        pool = [i for i, e in enumerate(data_storage.examples) if e[3] not in holdout]
        mi = int(getattr(args, "max_iters", 0))
        n = min(mi, len(data_storage)) if mi > 0 else len(data_storage)
        sampler = HoldoutTailSampler(hold, pool, n, generator)
        print(f"[holdout-tail] epoch당 {n} slice = hold-out {len(hold)} 전량 + "
              f"train 풀 {len(pool)} 중 {max(0, n - len(hold))} 무작위 (epoch마다 재추첨)")

    data_loader = DataLoader(
        dataset=data_storage,
        batch_size=args.batch_size,
        shuffle=(shuffle and sampler is None),   # sampler 와 shuffle 은 배타(DataLoader 가 거부)
        sampler=sampler,
        num_workers=num_workers,
        worker_init_fn=_worker_init_fn if num_workers > 0 else None,
        generator=generator,
        collate_fn=(_bbox_collate if return_bbox else None),
    )
    return data_loader


def _stratified_folds(ids_by_group, k, seed):
    """acc 그룹별 독립 셔플 → 각각 k등분 → fold별 합집합. 길이 k의 set 리스트 반환.

    각 그룹이 **자기만의** RandomState(seed)로 섞이므로 한 그룹의 fold 배정은
    다른 그룹의 존재 여부와 무관하다 — --acc 필터가 fold 경계를 못 움직이는 근거.
    그룹 순회는 sorted(키) 순서라 결정적이다.

    ⚠️ "독립"은 **다른 그룹의 영향을 안 받는다**는 뜻이지 순열이 서로 다르다는 뜻이 아니다.
    모든 그룹이 같은 seed를 쓰므로 크기가 같은 두 그룹은 **동일한 순열**을 받는다(정렬한
    리스트의 같은 위치가 같은 fold로 간다). 이건 결함이 아니다 — 결정성·균형·acc 불변식이
    모두 성립하고, 혹시 knee_acc4_i와 knee_acc8_i가 같은 출처의 볼륨이라면 오히려 둘을 같은
    fold에 묶어 누수를 막는다. 그러니 seed+hash(key) 같은 걸로 **탈상관시키지 말 것** —
    acc4_i와 acc8_i의 관계를 먼저 확인하지 않고 바꾸면 없던 누수를 만들 수 있다.
    """
    folds = [set() for _ in range(k)]
    for key in sorted(ids_by_group):
        ids = np.array(sorted(ids_by_group[key]))
        if len(ids) < k:
            raise ValueError(
                f"K-Fold 층화: acc{key} 볼륨 {len(ids)}개 < k={k} — 빈 fold가 생긴다.")
        shuffled = ids[np.random.RandomState(seed).permutation(len(ids))]
        for i, chunk in enumerate(np.array_split(shuffled, k)):
            folds[i].update(map(str, chunk))
    return folds


def kfold_roots(args):
    """K-Fold가 합쳐 쓰는 데이터 root 목록 (원본 train/ + val/ 전부 = 200 volume)."""
    return [Path(args.data_path_train), Path(args.data_path_val)]


def kfold_split(args):
    """K-Fold 분할 → (train_ids, val_ids). 원소는 volume_id(고유 볼륨 식별자).

    **분할 단위는 volume**이다. train/과 val/에 이름이 같은 볼륨이 30쌍 있지만 서로 남남이라
    (예: train/…/knee_acc4_1.h5 = 23슬라이스, val/…/knee_acc4_1.h5 = 31슬라이스) 각각 독립
    단위로 다룬다. fold 크기는 그룹별 k등분의 합이라, 그룹 볼륨 수가 k로 안 나눠떨어지면
    fold마다 ±1 차이가 난다(총합은 그대로).

    **acc 층화(2026-07-26~)**: acc4·acc8 볼륨을 _stratified_folds로 **각각** k등분해 fold별로
    합친다. 그룹별 셔플이 서로 독립이라 --acc all/4/8 중 뭘 줘도 **같은 볼륨은 항상 같은
    fold** → MoE의 pretrain(--acc all)→미세조정(--acc 4/8)을 같은 hold-out으로 이어도
    채점셋이 오염되지 않는다. (구버전은 "필터 후 분할"이라 경계가 --acc에 종속 —
    실측 40개 중 30개 오염. 그 코드는 이 층화로 대체됨.)
    ⚠️ 구버전(비층화)으로 나눈 기존 런(promptmr_kfold_f1/f2)과는 hold-out이 다르다.
    그 런들과 val 숫자를 나란히 비교하지 말 것.

    난수는 seed만의 함수(RandomState(kfold_seed))라 같은 seed면 언제 돌려도 같은 분할이 나온다.

    ⚠️ 동작 변경: 파일명에 accN 토큰이 없는 h5는 이제 ValueError로 즉시 중단시킨다. 층화는
    모든 볼륨을 어느 한 그룹에 넣어야 성립하기 때문이다. 구버전은 --acc all일 때 그런 파일을
    조용히 포함시켰으므로, 여태 돌던 설정이 갑자기 멈출 수 있다(이 데이터셋은 acc4/acc8 전수라
    정상 경로에선 안 걸린다).
    """
    k = int(getattr(args, "k_fold", 5))
    val_fold = int(getattr(args, "val_fold", 0))
    seed = int(getattr(args, "kfold_seed", 430))
    if not 0 <= val_fold < k:
        raise ValueError(f"--val-fold({val_fold})는 0 이상 --k-fold({k}) 미만이어야 합니다.")
    acc_want = _norm_acc(getattr(args, "acc", "all"))

    roots = kfold_roots(args)
    ids_by_group = {}   # acc 토큰('4'/'8') → volume_id 리스트
    no_token = []
    for r in roots:
        for f in sorted((r / "kspace").glob("*.h5")):
            acc = _file_acc(f.name)
            if acc is None:
                no_token.append(str(f))
                continue
            ids_by_group.setdefault(acc, []).append(volume_id(r, f.name))
    if no_token:
        raise ValueError(
            "K-Fold 층화: 파일명에 accN 토큰이 없어 그룹을 정할 수 없습니다:\n  "
            + "\n  ".join(no_token))
    if not ids_by_group:
        raise ValueError(f"K-Fold: 데이터 경로({roots})에서 h5 파일을 찾지 못했습니다.")

    folds = _stratified_folds(ids_by_group, k, seed)

    if acc_want is not None:
        # --acc 4/8: 조립된 fold에서 그 그룹만 남긴다. 경계 자체는 안 움직인다(층화 불변식).
        keep = set(ids_by_group.get(acc_want, []))
        if not keep:
            raise ValueError(f"K-Fold: acc={acc_want} 볼륨이 없습니다({roots}).")
        folds = [f & keep for f in folds]

    val_ids = folds[val_fold]
    train_ids = set().union(*folds) - val_ids
    return train_ids, val_ids


def create_kfold_data_loaders(args):
    """kfold_split의 분할로 (train_loader, val_loader) 생성. 두 로더 모두 train/·val/ 양쪽을 본다."""
    train_ids, val_ids = kfold_split(args)
    roots = kfold_roots(args)

    # --remask 를 쓰면 train 은 --acc all(160 vol) 로 넓히지만, **val 은 그 전문가의 가속률로 되돌린다.**
    #   안 되돌리면 acc4 전문가가 acc8 이 섞인 val 로 best_model 을 고르게 된다.
    #   층화 불변식(kfold_split 이 fold 를 조립한 **뒤** 필터) 덕분에 여기서 거른 val 집합은
    #   --acc 4 로 돌렸을 때의 val_ids 와 정확히 같다 → 과거 ft4/ft8 val 곡선과 비교 가능.
    remask = str(getattr(args, "remask", "off"))
    val_acc = None if remask in ("off", "") else remask

    # val 로더를 **먼저** 만든다 — --holdout-tail 의 hold-out 집합을 그 로더의 실제 구성에서
    #   그대로 꺼내려고(acc_override 필터까지 반영된 볼륨 목록 = 중복 판정 로직 0줄).
    #   순서를 되돌리면 holdout 이 빈 채로 train 로더가 만들어져 기능이 조용히 사라진다.
    #   val 로더는 그대로 hold-out 만 본다(계기판 역할은 tail 런에서 어차피 포기).
    val_loader = create_data_loaders(roots, args, shuffle=False, is_train=False, volumes=val_ids,
                                     acc_override=val_acc)
    holdout = ({e[3] for e in val_loader.dataset.examples}
               if getattr(args, "holdout_tail", False) else None)
    train_loader = create_data_loaders(roots, args, shuffle=True, is_train=True,
                                       volumes=(train_ids | holdout) if holdout else train_ids,
                                       holdout=holdout)

    def _vols(loader):
        """로그를 분할집합이 아니라 **로더의 실제 구성**에서 뽑는다 — acc_override 로 val 이
        걸러지면 train_ids/val_ids 는 더 이상 로더 내용과 같지 않다."""
        return {e[3] for e in loader.dataset.examples}

    def _n(ids, tok):
        return sum(1 for v in ids if tok in v)

    tr_v, va_v = _vols(train_loader), _vols(val_loader)
    acc = getattr(args, "acc", "all")
    print(f"[K-Fold] K={args.k_fold} · val-fold={args.val_fold} · seed={getattr(args, 'kfold_seed', 430)} "
          f"· 분할 단위=volume · acc={acc}"
          + (f" · remask={remask}(val 은 acc{val_acc})" if val_acc else "")
          + f" (전체 {len(train_ids) + len(val_ids)} volume)")
    print(f"[K-Fold] train {len(tr_v):3d} vol (acc4 {_n(tr_v, 'acc4')} · acc8 {_n(tr_v, 'acc8')}) "
          f"· {len(train_loader.dataset)} slice")
    print(f"[K-Fold] val   {len(va_v):3d} vol (acc4 {_n(va_v, 'acc4')} · acc8 {_n(va_v, 'acc8')}) "
          f"· {len(val_loader.dataset)} slice")

    return train_loader, val_loader
