"""affine 증강의 target 재생성 규약(_rss_crop)이 제공 label과 정합하는지 실데이터로 확인.

비GPU — 서버 분할 터미널에서 바로 실행한다 (remask의 280볼륨 검증 선례):
    python scripts/verify_affine_target.py --data /root/Data/train
검증 1 (규약): 워프 없이 full k-space → _rss_crop == image_label (corr ≥ 0.9999).
    실패하면 crop/pad 규약이 주최측 label 생성과 다른 것 — 학습 전에 반드시 잡아야 한다.
검증 2 (비용): _affine 1회 시간(ms/slice) 실측 — p=0.5면 평균 부하는 그 절반.
"""
import argparse
import os
import sys
import time

import h5py
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.data.transforms import AugmentConfig, MRIAugmentor, _ifftc, _rss_crop  # noqa: E402


class _FixedRng:
    def __init__(self, deg, s):
        self.vals = [deg, s]
    def uniform(self, a, b):
        return self.vals.pop(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/root/Data/train", help=".../train (아래 image/·kspace/ 전제)")
    ap.add_argument("--max-volumes", type=int, default=0, help="0=전수")
    args = ap.parse_args()

    kdir, idir = os.path.join(args.data, "kspace"), os.path.join(args.data, "image")
    files = sorted(f for f in os.listdir(kdir) if f.endswith(".h5"))
    if args.max_volumes:
        files = files[: args.max_volumes]

    aug = MRIAugmentor(AugmentConfig(on=True))
    bad, times = [], []
    for i, fn in enumerate(files):
        with h5py.File(os.path.join(kdir, fn), "r") as fk, \
             h5py.File(os.path.join(idir, fn), "r") as fi:
            sl = fk["kspace"].shape[0] // 2
            kspace = np.asarray(fk["kspace"][sl])          # (coils, H, W) complex, full
            label = np.asarray(fi["image_label"][sl], dtype=np.float32)
        regen = _rss_crop(_ifftc(kspace), label.shape)
        corr = float(np.corrcoef(regen.ravel(), label.ravel())[0, 1])
        t0 = time.perf_counter()
        aug._affine(kspace, label.shape, _FixedRng(5.0, 1.1))
        times.append((time.perf_counter() - t0) * 1e3)
        mark = "OK " if corr >= 0.9999 else "BAD"
        if corr < 0.9999:
            bad.append((fn, corr))
        print(f"[{i+1}/{len(files)}] {mark} {fn}  corr={corr:.10f}")

    print(f"\n_affine 시간: median {np.median(times):.1f} ms/slice (p=0.5면 평균 부하는 절반)")
    if bad:
        print(f"[FAIL] corr<0.9999 볼륨 {len(bad)}개 — _rss_crop 규약이 label 생성과 다르다. 학습 금지.")
        for fn, c in bad:
            print(f"    {fn}  corr={c:.10f}")
        sys.exit(1)
    print("[PASS] 재생성 규약이 전 볼륨에서 label과 정합.")


if __name__ == "__main__":
    main()
