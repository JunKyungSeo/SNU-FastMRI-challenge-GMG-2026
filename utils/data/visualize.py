"""
FastMRI 2026 데이터 시각화 스크립트.

utils/data 폴더에 있는 단일 샘플 h5 파일을 시각화한다.
  - 이미지 파일 (img_*.h5) : image_input / image_grappa / image_label + bbox annotation
  - k-space 파일 (kspace_*.h5): 코일별/코일결합 k-space 로그 크기, 언더샘플 mask

사용 예시:
    # 파일 구조/annotation 요약만 출력
    python utils/data/visualize.py --info

    # acc4 이미지의 13번 슬라이스(병변 있는 슬라이스) 시각화 후 PNG 저장
    python utils/data/visualize.py --acc 4 --slice 13 --save

    # k-space + mask 함께 보기
    python utils/data/visualize.py --acc 8 --slice 15 --kspace --save
"""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

# matplotlib 은 시각화 시점에만 필요 → --info 는 미설치 환경에서도 동작하도록 지연 임포트

# 이 스크립트가 있는 폴더(= utils/data)를 데이터 폴더로 사용
DATA_DIR = Path(__file__).resolve().parent

IMAGE_KEYS = ("image_input", "image_grappa", "image_label")


def _paths(acc):
    """가속률(acc)에 해당하는 (이미지 파일, k-space 파일) 경로를 찾는다."""
    img = sorted(DATA_DIR.glob(f"img*acc{acc}_*.h5"))
    ksp = sorted(DATA_DIR.glob(f"kspace*acc{acc}_*.h5"))
    if not img or not ksp:
        raise FileNotFoundError(
            f"acc{acc} 파일을 찾지 못했습니다. "
            f"img: {[p.name for p in img]}, kspace: {[p.name for p in ksp]}"
        )
    return img[0], ksp[0]


def _load_annotations(hf):
    """이미지 h5 attrs 안의 annotation JSON을 {slice_idx(int): [bbox,...]} 로 파싱."""
    raw = hf.attrs.get("annotations", None)
    if raw is None:
        return {}
    ann = json.loads(raw)
    return {int(k): v for k, v in ann.items()}


def print_info():
    """폴더 내 모든 h5 파일의 키/shape/attrs 요약을 출력."""
    files = sorted(DATA_DIR.glob("*.h5"))
    if not files:
        print(f"[info] {DATA_DIR} 에 h5 파일이 없습니다.")
        return
    for f in files:
        print("=" * 70)
        print(f.name)
        with h5py.File(f, "r") as hf:
            for k in hf.keys():
                d = hf[k]
                print(f"  [{k}] shape={d.shape} dtype={d.dtype}")
            for ak, av in hf.attrs.items():
                if ak == "annotations":
                    ann = _load_annotations(hf)
                    print(f"  (attr) annotations: {len(ann)}개 슬라이스에 bbox 존재")
                    for s in sorted(ann):
                        labels = [b["label"] for b in ann[s]]
                        print(f"        slice {s}: {labels}")
                else:
                    print(f"  (attr) {ak}: {av}")


def _draw_bboxes(ax, bboxes, color="red"):
    """annotation bbox(레드박스)를 축에 그린다."""
    from matplotlib.patches import Rectangle
    for b in bboxes:
        rect = Rectangle(
            (b["x"], b["y"]), b["width"], b["height"],
            linewidth=1.5, edgecolor=color, facecolor="none",
        )
        ax.add_patch(rect)
        ax.text(
            b["x"], b["y"] - 3, b["label"],
            color=color, fontsize=6, va="bottom",
            bbox=dict(facecolor="black", alpha=0.4, pad=0.5, edgecolor="none"),
        )


def visualize_image(acc, slice_idx, save=False):
    """image_input / image_grappa / image_label 을 나란히 그리고 bbox 오버레이."""
    import matplotlib.pyplot as plt
    img_path, _ = _paths(acc)
    with h5py.File(img_path, "r") as hf:
        n_slices = hf["image_label"].shape[0]
        if not 0 <= slice_idx < n_slices:
            raise IndexError(f"slice {slice_idx} 범위 밖 (0~{n_slices - 1})")
        imgs = {k: hf[k][slice_idx] for k in IMAGE_KEYS}
        ann = _load_annotations(hf)
    bboxes = ann.get(slice_idx, [])

    fig, axes = plt.subplots(1, len(IMAGE_KEYS), figsize=(4 * len(IMAGE_KEYS), 4.5))
    for ax, key in zip(axes, IMAGE_KEYS):
        ax.imshow(imgs[key], cmap="gray")
        title = key.replace("image_", "")
        ax.set_title(f"{title}", fontsize=11)
        ax.axis("off")
        _draw_bboxes(ax, bboxes)

    n_box = len(bboxes)
    fig.suptitle(
        f"{img_path.name} | slice {slice_idx}/{n_slices - 1} | {n_box} bbox",
        fontsize=12,
    )
    fig.tight_layout()
    _finish(fig, save, f"image_acc{acc}_slice{slice_idx}")


def visualize_kspace(acc, slice_idx, save=False):
    """k-space 로그 크기(코일 결합)와 언더샘플 mask 시각화."""
    import matplotlib.pyplot as plt
    _, ksp_path = _paths(acc)
    with h5py.File(ksp_path, "r") as hf:
        n_slices = hf["kspace"].shape[0]
        if not 0 <= slice_idx < n_slices:
            raise IndexError(f"slice {slice_idx} 범위 밖 (0~{n_slices - 1})")
        kspace = hf["kspace"][slice_idx]          # (coils, H, W) complex
        mask = np.asarray(hf["mask"])             # (W,)

    # 코일 방향 RSS(root-sum-square)로 결합한 크기 스펙트럼
    kmag = np.sqrt(np.sum(np.abs(kspace) ** 2, axis=0))
    klog = np.log(kmag + 1e-9)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), gridspec_kw={"width_ratios": [3, 1]})
    axes[0].imshow(klog, cmap="gray", aspect="auto")
    axes[0].set_title(f"k-space log|·| (RSS over {kspace.shape[0]} coils)")
    axes[0].axis("off")

    axes[1].imshow(mask[None, :].repeat(40, axis=0), cmap="gray", aspect="auto")
    sampled = int(mask.sum())
    axes[1].set_title(f"mask: {sampled}/{mask.size} lines")
    axes[1].axis("off")

    fig.suptitle(f"{ksp_path.name} | slice {slice_idx}/{n_slices - 1}", fontsize=12)
    fig.tight_layout()
    _finish(fig, save, f"kspace_acc{acc}_slice{slice_idx}")


def _finish(fig, save, stem):
    import matplotlib.pyplot as plt
    if save:
        out = DATA_DIR / f"viz_{stem}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"[saved] {out}")
    else:
        plt.show()
    plt.close(fig)


def parse():
    p = argparse.ArgumentParser(description="FastMRI 2026 데이터 시각화")
    p.add_argument("--info", action="store_true", help="파일 구조/annotation 요약만 출력")
    p.add_argument("--acc", type=int, default=4, choices=[4, 8], help="가속률 (4 또는 8)")
    p.add_argument("--slice", type=int, default=None, help="시각화할 슬라이스 인덱스")
    p.add_argument("--kspace", action="store_true", help="k-space+mask도 함께 시각화")
    p.add_argument("--save", action="store_true", help="화면 표시 대신 PNG로 저장")
    return p.parse_args()


def main():
    args = parse()
    if args.info:
        print_info()
        return

    slice_idx = args.slice
    if slice_idx is None:
        # 슬라이스 미지정 시 annotation(병변)이 있는 첫 슬라이스를 자동 선택
        img_path, _ = _paths(args.acc)
        with h5py.File(img_path, "r") as hf:
            ann = _load_annotations(hf)
            n_slices = hf["image_label"].shape[0]
        slice_idx = min(ann) if ann else n_slices // 2
        print(f"[auto] slice 미지정 → slice {slice_idx} 선택")

    visualize_image(args.acc, slice_idx, save=args.save)
    if args.kspace:
        visualize_kspace(args.acc, slice_idx, save=args.save)


if __name__ == "__main__":
    main()
