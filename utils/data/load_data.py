import h5py
import random
import re
from utils.data.transforms import DataTransform
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import numpy as np


def _acc_key(name):
    """파일명에서 'acc4_1', 'acc8_1' 같은 (가속률_인덱스) 토큰을 뽑아 페어링 키로 사용.
    이미지(img_...)와 k-space(kspace...) 파일명이 달라도 같은 샘플끼리 짝지을 수 있게 한다."""
    m = re.search(r'(acc\d+_\d+)', name)
    return m.group(1) if m else name


class SliceData(Dataset):
    def __init__(self, root, transform, input_key, target_key, forward=False):
        self.transform = transform
        self.input_key = input_key
        self.target_key = target_key
        self.forward = forward
        self.image_examples = []
        self.kspace_examples = []

        # ------------------------------------------------------------------
        # [원본] root/image, root/kspace 하위 폴더 스캔 방식 (전체 데이터셋용)
        # 아직 데이터셋이 준비되지 않아 임시로 주석 처리.
        # ------------------------------------------------------------------
        # if not forward:
        #     image_files = list(Path(root / "image").iterdir())
        #     for fname in sorted(image_files):
        #         num_slices = self._get_metadata(fname)
        #
        #         self.image_examples += [
        #             (fname, slice_ind) for slice_ind in range(num_slices)
        #         ]
        #
        # kspace_files = list(Path(root / "kspace").iterdir())
        # for fname in sorted(kspace_files):
        #     num_slices = self._get_metadata(fname)
        #
        #     self.kspace_examples += [
        #         (fname, slice_ind) for slice_ind in range(num_slices)
        #     ]

        # ------------------------------------------------------------------
        # [임시] utils/data 폴더에 평평하게 놓인 단일 샘플 파일 직접 로딩.
        #   이미지(타깃) : img_knee_acc4_1.h5, img_knee_acc8_1.h5
        #   k-space(입력): kspace_knee_acc4_1.h5, kspaceimg_acc8_1.h5
        # acc 토큰(acc4_1 / acc8_1)으로 이미지·k-space를 짝짓는다.
        # 인자로 받은 root(data_path) 대신 이 파일이 있는 utils/data 폴더를 사용한다.
        # ------------------------------------------------------------------
        root = Path(__file__).resolve().parent
        kspace_files = sorted(root.glob("kspace*.h5"))

        image_map = {}
        if not forward:
            image_files = sorted(root.glob("img*.h5"))
            image_map = {_acc_key(f.name): f for f in image_files}

        for kspace_fname in kspace_files:
            num_slices = self._get_metadata(kspace_fname)
            self.kspace_examples += [
                (kspace_fname, slice_ind) for slice_ind in range(num_slices)
            ]
            if not forward:
                key = _acc_key(kspace_fname.name)
                if key not in image_map:
                    raise ValueError(
                        f"k-space 파일 {kspace_fname.name} 에 대응하는 이미지 파일을 "
                        f"data 폴더에서 찾지 못했습니다 (키: {key})."
                    )
                image_fname = image_map[key]
                self.image_examples += [
                    (image_fname, slice_ind) for slice_ind in range(num_slices)
                ]


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
        if not self.forward and _acc_key(image_fname.name) != _acc_key(kspace_fname.name):
            raise ValueError(f"Image file {image_fname.name} does not match kspace file {kspace_fname.name}")

        with h5py.File(kspace_fname, "r") as hf:
            input = hf[self.input_key][dataslice]
            mask =  np.array(hf["mask"])
        if self.forward:
            target = -1
            attrs = -1
        else:
            with h5py.File(image_fname, "r") as hf:
                target = hf[self.target_key][dataslice]
                attrs = dict(hf.attrs)
            
        return self.transform(mask, input, target, attrs, kspace_fname.name, dataslice)


def create_data_loaders(data_path, args, shuffle=False, isforward=False):
    if isforward == False:
        max_key_ = args.max_key
        target_key_ = args.target_key
    else:
        max_key_ = -1
        target_key_ = -1
    data_storage = SliceData(
        root=data_path,
        transform=DataTransform(isforward, max_key_),
        input_key=args.input_key,
        target_key=target_key_,
        forward = isforward
    )

    data_loader = DataLoader(
        dataset=data_storage,
        batch_size=args.batch_size,
        shuffle=shuffle,
    )
    return data_loader
