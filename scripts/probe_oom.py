"""8GB 레버 스윕용 OOM 탐침.

각 (cascades, channels) 조합으로 합성 데이터 1 forward+backward를 돌려
peak CUDA 메모리 또는 OOM을 출력한다. GPU에서 실행:

    python scripts/probe_oom.py --cascades 6 --channels 24 48
"""
import argparse
import os
import sys

# repo 루트를 path에 추가(어디서 실행하든 utils.* import 가능)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402
from utils.model.sdum.config import tiny_8gb_args  # noqa: E402
from utils.model.sdum.adapter import build_sdum  # noqa: E402


def probe(cascades, channels, H=384, W=372, C=15):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    args = tiny_8gb_args(num_cascades=cascades, channels=channels)
    model = build_sdum(args).to(dev).train()
    ksp = torch.randn(1, C, H, W, 2, device=dev)
    mask = torch.zeros(1, 1, 1, W, 1, dtype=torch.uint8, device=dev)
    mask[..., ::4, :] = 1
    mask[..., W // 2 - 15 : W // 2 + 15, :] = 1
    tgt = torch.randn(1, 384, 384, device=dev)
    if dev == "cuda":
        torch.cuda.reset_peak_memory_stats()
    try:
        out = model(ksp, mask)
        loss = torch.nn.functional.l1_loss(out, tgt)
        loss.backward()
        if dev == "cuda":
            peak = torch.cuda.max_memory_allocated() / 1024**3
            print(f"OK cascades={cascades} channels={channels} peak={peak:.2f}GB")
        else:
            print(f"OK (cpu) cascades={cascades} channels={channels}")
    except torch.cuda.OutOfMemoryError:
        print(f"OOM cascades={cascades} channels={channels}")
        torch.cuda.empty_cache()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cascades", type=int, default=4)
    ap.add_argument("--channels", type=int, nargs="+", default=[32, 64])
    args = ap.parse_args()
    probe(args.cascades, args.channels)
