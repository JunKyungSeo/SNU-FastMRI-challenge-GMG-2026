"""CLI: build a summary PNG for image-format fastMRI h5 file(s).

For each file it picks 3 evenly-spaced slices (12 slices -> 4, 8, 12), shows
Label | Input | GRAPPA for each (bbox drawn on Label), and prints the file
attrs at the bottom. Output goes to  <repo root>/viewer_result/<name>_summary.png

Usage:
    python scripts/summarize.py file1.h5 [file2.h5 ...]
    python scripts/summarize.py         # summarize every *.h5 in current dir
Deps: pip install h5py numpy matplotlib
"""

import argparse
import json
import os
import sys

import h5py
import numpy as np
import matplotlib

matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt
import matplotlib.patches as patches

IMAGE_KEYS = ("image_label", "image_input", "image_grappa")
COL_TITLES = ("Label", "Input", "GRAPPA")
OUT_DIRNAME = "viewer_result"


def pick_slices(n, annotations=None, count=3):
    """Pick `count` slices to summarize.

    Priority:
      1. Annotated slices first, chosen greedily to cover as many *distinct*
         bbox labels as possible (maximize variety of annotations).
      2. Any remaining annotated slices.
      3. Fill leftover slots with evenly-spaced slices
         (n=12, no annotations -> slices 4, 8, 12 == 0-based [3, 7, 11]).
    Result is returned sorted ascending for display.
    """
    annotations = annotations or {}

    def labels_of(idx):
        return {str(b.get("label", "")).strip() for b in annotations.get(idx, [])}

    picks = []
    covered = set()
    remaining = sorted(k for k in annotations if 0 <= k < n and annotations[k])
    while remaining and len(picks) < count:
        # most new labels wins; tie -> earliest slice (larger -i)
        best = max(remaining, key=lambda i: (len(labels_of(i) - covered), -i))
        picks.append(best)
        covered |= labels_of(best)
        remaining.remove(best)

    if len(picks) < count:
        for i in range(1, count + 1):
            idx = min(max(round(i * n / count) - 1, 0), n - 1)
            if idx not in picks:
                picks.append(idx)
            if len(picks) >= count:
                break

    return sorted(dict.fromkeys(picks))[:count]


def load_annotations(attrs):
    raw = attrs.get("annotations")
    if raw is None:
        return {}
    try:
        return {int(k): v for k, v in json.loads(raw).items()}
    except (json.JSONDecodeError, ValueError, AttributeError, TypeError):
        return {}


def attrs_to_text(attrs):
    if not attrs:
        return "(no attrs)"
    lines = []
    for key, value in attrs.items():
        if key == "annotations":
            try:
                value = json.dumps(json.loads(value), ensure_ascii=False)
            except (json.JSONDecodeError, TypeError):
                pass
        lines.append(f"{key}: {value}")
    return "\n".join(lines)


def summarize_file(path, out_dir):
    with h5py.File(path, "r") as f:
        missing = [k for k in IMAGE_KEYS if k not in f]
        if missing:
            print(f"[skip] {os.path.basename(path)}: not an image h5 "
                  f"(missing {', '.join(missing)}; keys={list(f.keys())})")
            return None
        volumes = {k: f[k][:] for k in IMAGE_KEYS}
        attrs = {k: f.attrs[k] for k in f.attrs.keys()}

    annotations = load_annotations(attrs)
    n = volumes["image_label"].shape[0]
    slices = pick_slices(n, annotations, 3)

    rows = len(slices)
    fig = plt.figure(figsize=(9, 3 * rows + 1.6))
    gs = fig.add_gridspec(rows + 1, 3, height_ratios=[3] * rows + [1.1])
    fig.suptitle(os.path.basename(path), fontsize=13, fontweight="bold")

    for r, sidx in enumerate(slices):
        boxes = annotations.get(sidx, [])
        for c, (key, title) in enumerate(zip(IMAGE_KEYS, COL_TITLES)):
            ax = fig.add_subplot(gs[r, c])
            ax.imshow(volumes[key][sidx], cmap="gray")
            ax.axis("off")
            extra = f"  [{len(boxes)} bbox]" if (key == "image_label" and boxes) else ""
            ax.set_title(f"{title} — slice {sidx}{extra}", fontsize=9)
            if key == "image_label":
                for box in boxes:
                    ax.add_patch(patches.Rectangle(
                        (box["x"], box["y"]), box["width"], box["height"],
                        linewidth=1.5, edgecolor="red", facecolor="none"))
                    ax.text(box["x"], box["y"] - 3, str(box.get("label", "")).strip(),
                            color="red", fontsize=7,
                            bbox=dict(facecolor="black", alpha=0.5, pad=1, edgecolor="none"))

    ax_txt = fig.add_subplot(gs[rows, :])
    ax_txt.axis("off")
    ax_txt.text(0.0, 1.0, "Attributes\n" + attrs_to_text(attrs),
                va="top", ha="left", fontsize=8, family="monospace", wrap=True)

    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(path))[0]
    out_path = os.path.join(out_dir, f"{stem}_summary.png")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"[ok]   {os.path.basename(path)} -> {out_path}  (slices {slices})")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Build summary PNGs for image-format h5 files.")
    parser.add_argument("files", nargs="*", help="h5 file(s). Default: all *.h5 in current dir.")
    args = parser.parse_args()

    files = args.files or [f for f in os.listdir(".") if f.lower().endswith(".h5")]
    if not files:
        print("h5 파일을 찾을 수 없습니다. 파일 경로를 인자로 주세요.")
        sys.exit(1)

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = os.path.join(repo_root, OUT_DIRNAME)
    made = 0
    for path in files:
        if not os.path.isfile(path):
            print(f"[skip] {path}: 파일 없음")
            continue
        try:
            if summarize_file(path, out_dir):
                made += 1
        except OSError as e:
            print(f"[skip] {path}: 열기 실패 ({e})")
    print(f"\n완료: {made}개 요약 이미지 생성 -> {out_dir}")


if __name__ == "__main__":
    main()
