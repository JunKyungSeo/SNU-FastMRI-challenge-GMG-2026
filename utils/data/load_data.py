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


class SliceData(Dataset):
    def __init__(self, root, transform, input_key, target_key, forward=False, file_names=None):
        self.transform = transform
        self.input_key = input_key
        self.target_key = target_key
        self.forward = forward
        self.image_examples = []
        self.kspace_examples = []

        roots = root if isinstance(root, (list, tuple)) else [root]
        file_set = set(file_names) if file_names is not None else None

        for r in roots:
            r = Path(r)
            if not forward:
                image_dir = r / "image"
                if image_dir.exists():
                    for fname in sorted(list(image_dir.iterdir())):
                        if file_set is not None and fname.name not in file_set:
                            continue
                        num_slices = self._get_metadata(fname)
                        self.image_examples += [
                            (fname, slice_ind) for slice_ind in range(num_slices)
                        ]

            kspace_dir = r / "kspace"
            if kspace_dir.exists():
                for fname in sorted(list(kspace_dir.iterdir())):
                    if file_set is not None and fname.name not in file_set:
                        continue
                    num_slices = self._get_metadata(fname)
                    self.kspace_examples += [
                        (fname, slice_ind) for slice_ind in range(num_slices)
                    ]

        # ------------------------------------------------------------------
        # [임시·주석] utils/data 폴더에 평평하게 놓인 단일 샘플 파일 직접 로딩.
        #   데이터셋 미연결(로컬 테스트) 환경 전용. 원본 데이터셋 연결 시 위 [원본] 블록 사용.
        #   이미지(타깃) : img_knee_acc4_1.h5, img_knee_acc8_1.h5
        #   k-space(입력): kspace_knee_acc4_1.h5, kspace_knee_acc8_1.h5
        # acc 토큰(acc4_1 / acc8_1)으로 이미지·k-space를 짝짓는다(파일명이 달라도 페어링).
        # 인자로 받은 root(data_path) 대신 이 파일이 있는 utils/data 폴더를 사용한다.
        # ⚠️ 이 경로를 되살릴 땐 아래 __getitem__의 이름 비교도 _acc_key 버전으로 함께 교체할 것.
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
        return len(self.kspace_examples)

    def __getitem__(self, i):
        if not self.forward:
            image_fname, _ = self.image_examples[i]
        kspace_fname, dataslice = self.kspace_examples[i]
        if not self.forward and image_fname.name != kspace_fname.name:
            raise ValueError(f"Image file {image_fname.name} does not match kspace file {kspace_fname.name}")

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

        return self.transform(mask, input, target, attrs, kspace_fname.name, dataslice, grappa)


def create_data_loaders(data_path, args, shuffle=False, isforward=False, is_train=None, file_names=None):
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
    # 즉 train.py에 --aug-* 인자를 아직 안 넣었다면 기존 학습과 100% 동일하게 동작.
    # (인자 정의·매 epoch set_epoch 배선은 strategy/todo.md의 3·4번.)
    # ------------------------------------------------------------------
    aug_on = getattr(args, "aug_on", False)
    cfg = AugmentConfig(
        on=aug_on,
        schedule=getattr(args, "aug_schedule", "exp"),
        warmup_epochs=getattr(args, "aug_warmup_epochs", 10),
        ramp_epochs=getattr(args, "aug_ramp_epochs", 15),
        max_prob=getattr(args, "aug_max_prob", 0.5),
        flip=getattr(args, "aug_flip", True),
        mask=getattr(args, "aug_mask", True),
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
        file_names=file_names,
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


def create_kfold_data_loaders(args):
    """
    args.data_path_train 및 args.data_path_val 경로 하위의 모든 h5 Volume 파일들을 수집 후,
    args.k_fold 개수로 Volume 단위 K-Fold 분할을 수행합니다.
    args.val_fold 인덱스에 해당하는 Fold를 val_loader로, 나머지 Fold들을 train_loader로 생성합니다.
    """
    k = getattr(args, "k_fold", 5)
    val_fold = getattr(args, "val_fold", 0)
    seed = getattr(args, "kfold_seed", 430)

    train_path = Path(args.data_path_train)
    val_path = Path(args.data_path_val)
    roots = [train_path, val_path]

    # 모든 volume 파일명 (kspace 파일명 기준) 수집
    all_fnames = set()
    for r in roots:
        kspace_dir = r / "kspace"
        if kspace_dir.exists():
            for f in kspace_dir.glob("*.h5"):
                all_fnames.add(f.name)

    sorted_fnames = np.array(sorted(list(all_fnames)))
    if len(sorted_fnames) == 0:
        raise ValueError(f"K-Fold: 데이터 경로({train_path}, {val_path})에서 h5 파일을 찾지 못했습니다.")

    # Volume 단위 Deterministic Split
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(sorted_fnames))
    shuffled_fnames = sorted_fnames[perm]

    folds = np.array_split(shuffled_fnames, k)
    if val_fold < 0 or val_fold >= k:
        raise ValueError(f"val_fold ({val_fold})는 0 이상 {k} 미만이어야 합니다.")

    val_files = set(folds[val_fold])
    train_files = set(shuffled_fnames) - val_files

    print(f"[K-Fold] 총 Volume 파일 수: {len(sorted_fnames)}개 | K={k}, Val Fold={val_fold}")
    print(f"[K-Fold] Train Volume 수: {len(train_files)}개 | Val Volume 수: {len(val_files)}개")

    train_loader = create_data_loaders(roots, args, shuffle=True, is_train=True, file_names=train_files)
    val_loader = create_data_loaders(roots, args, shuffle=False, is_train=False, file_names=val_files)

    return train_loader, val_loader
