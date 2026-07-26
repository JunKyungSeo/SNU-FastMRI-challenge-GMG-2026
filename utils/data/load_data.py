import h5py
import random
import re
import torch
from utils.data.transforms import DataTransform, MRIAugmentor, AugmentConfig
from torch.utils.data import Dataset, DataLoader
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
    def __init__(self, root, transform, input_key, target_key, forward=False, volumes=None, acc=None):
        self.transform = transform
        self.input_key = input_key
        self.target_key = target_key
        self.forward = forward
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
            input = hf[self.input_key][dataslice]
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


def create_data_loaders(data_path, args, shuffle=False, isforward=False, is_train=None, volumes=None):
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
    # augmentation 설정 — train.py args에서 읽음(없으면 getattr 기본값 = 전부 off).
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
        flip=getattr(args, "aug_flip", True),
        mask=getattr(args, "aug_mask", True),
        weight_flip=getattr(args, "aug_weight_flip", 1.0),
        weight_mask=getattr(args, "aug_weight_mask", 1.0),
        phase=getattr(args, "aug_phase", False),
        noise=getattr(args, "aug_noise", False),
        mask_thin_max=getattr(args, "aug_mask_thin_max", 4),
        orig_mask_prob=getattr(args, "aug_orig_mask_prob", 0.5),
        noise_sigma=getattr(args, "aug_noise_sigma", 0.0),
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
    n_virtual_coils = getattr(args, "n_virtual_coils", 8)
    # B5: 손실·검증을 채점 정합 foreground 영역으로 제한(기본 off). train/val 양쪽 로더에 부착
    #     (forward는 target 없어 DataTransform이 내부적으로 미포함) → 소비부 train_epoch/validate 확장.
    fg_loss = getattr(args, "fg_loss", False)

    data_storage = SliceData(
        root=data_path,
        transform=DataTransform(
            isforward, max_key_,
            augmentor=augmentor, is_train=is_train, return_bbox=return_bbox,
            normalize=normalize, readout_crop=readout_crop, crop_h=crop_h,
            norm_percentile=norm_percentile, return_grappa=return_grappa,
            coil_compress=coil_compress, n_virtual_coils=n_virtual_coils,
            return_fg_mask=fg_loss,
        ),
        input_key=args.input_key,
        target_key=target_key_,
        forward=isforward,
        volumes=volumes,
        acc=getattr(args, "acc", "all"),
    )

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

    data_loader = DataLoader(
        dataset=data_storage,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        worker_init_fn=_worker_init_fn if num_workers > 0 else None,
        generator=generator,
        collate_fn=(_bbox_collate if return_bbox else None),
    )
    return data_loader


def kfold_roots(args):
    """K-Fold가 합쳐 쓰는 데이터 root 목록 (원본 train/ + val/ 전부 = 200 volume)."""
    return [Path(args.data_path_train), Path(args.data_path_val)]


def kfold_split(args):
    """K-Fold 분할 → (train_ids, val_ids). 원소는 volume_id(고유 볼륨 식별자).

    **분할 단위는 volume**이다. train/과 val/에 이름이 같은 볼륨이 30쌍 있지만 서로 남남이라
    (예: train/…/knee_acc4_1.h5 = 23슬라이스, val/…/knee_acc4_1.h5 = 31슬라이스) 각각 독립
    단위로 다룬다 → fold마다 정확히 200/k개가 떨어진다.

    ⚠️ --acc 4/8 을 함께 주면 그 가속률 볼륨만 모아서 분할한다. 즉 **--acc 값이 바뀌면 fold
    경계도 같이 바뀐다.** --acc 하나만 쓰는 단독 실험에선 정상이지만, 한 파이프라인 안에서
    --acc를 바꿔가며(MoE의 pretrain --acc all → 미세조정 --acc 4/8) 같은 hold-out으로 채점하려
    하면 단계마다 빼두는 볼륨이 달라져 채점셋이 오염된다(실측: 40개 중 30개). 그 조합을 쓸
    거면 acc4·acc8을 각각 k등분해 합치는 층화 분할로 먼저 바꿔야 한다.

    난수는 seed만의 함수(RandomState(kfold_seed))라 같은 seed면 언제 돌려도 같은 분할이 나온다.
    """
    k = int(getattr(args, "k_fold", 5))
    val_fold = int(getattr(args, "val_fold", 0))
    seed = int(getattr(args, "kfold_seed", 430))
    if not 0 <= val_fold < k:
        raise ValueError(f"--val-fold({val_fold})는 0 이상 --k-fold({k}) 미만이어야 합니다.")
    acc_want = _norm_acc(getattr(args, "acc", "all"))

    roots = kfold_roots(args)
    ids = []
    for r in roots:
        for f in sorted((r / "kspace").glob("*.h5")):
            if acc_want is not None and _file_acc(f.name) != acc_want:
                continue
            ids.append(volume_id(r, f.name))
    if not ids:
        raise ValueError(f"K-Fold: 데이터 경로({roots})에서 h5 파일을 찾지 못했습니다"
                         f"{'' if acc_want is None else f' (acc={acc_want} 필터 적용)'}.")

    ids = np.array(sorted(ids))
    shuffled = ids[np.random.RandomState(seed).permutation(len(ids))]
    val_ids = set(map(str, np.array_split(shuffled, k)[val_fold]))
    train_ids = set(map(str, shuffled)) - val_ids
    return train_ids, val_ids


def create_kfold_data_loaders(args):
    """kfold_split의 분할로 (train_loader, val_loader) 생성. 두 로더 모두 train/·val/ 양쪽을 본다."""
    train_ids, val_ids = kfold_split(args)
    roots = kfold_roots(args)

    train_loader = create_data_loaders(roots, args, shuffle=True, is_train=True, volumes=train_ids)
    val_loader = create_data_loaders(roots, args, shuffle=False, is_train=False, volumes=val_ids)

    def _n(ids, tok):
        return sum(1 for v in ids if tok in v)

    acc = getattr(args, "acc", "all")
    print(f"[K-Fold] K={args.k_fold} · val-fold={args.val_fold} · seed={getattr(args, 'kfold_seed', 430)} "
          f"· 분할 단위=volume · acc={acc} (전체 {len(train_ids) + len(val_ids)} volume)")
    print(f"[K-Fold] train {len(train_ids):3d} vol (acc4 {_n(train_ids, 'acc4')} · acc8 {_n(train_ids, 'acc8')}) "
          f"· {len(train_loader.dataset)} slice")
    print(f"[K-Fold] val   {len(val_ids):3d} vol (acc4 {_n(val_ids, 'acc4')} · acc8 {_n(val_ids, 'acc8')}) "
          f"· {len(val_loader.dataset)} slice")

    return train_loader, val_loader
