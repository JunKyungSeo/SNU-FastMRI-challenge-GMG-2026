"""PromptMR+ 8GB 레버 스윕용 OOM 탐침 (probe_oom.py의 promptmr 버전).

실제 학습 입력 shape(readout-crop 없음 → full H=640)로 1 forward+backward(train 모드,
checkpointing 활성)를 돌려 peak CUDA 메모리 또는 OOM을 한 줄로 출력한다. 프로세스당 1 config
(VRAM 깨끗이 초기화). GPU에서 실행:

    python scripts/probe_oom_promptmr.py --num-cascades 10 --n-feat0 48
"""
import argparse
import os
import sys
from types import SimpleNamespace

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(1, os.path.join(_ROOT, "utils", "model"))  # train.py처럼: utils/model/fastmri 를 top-level로

import torch  # noqa: E402
from utils.model.promptmr.adapter import build_promptmr  # noqa: E402


def probe(nc, nf, snf, nh, ckpt, H=640, W=368, C=15):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    args = SimpleNamespace(num_cascades=nc, n_feat0=nf, sens_n_feat0=snf,
                           n_history=nh, use_checkpoint=ckpt)
    tag = f"cascades={nc} n_feat0={nf} sens={snf} n_hist={nh} ckpt={ckpt} coils={C} HxW={H}x{W}"
    model = build_promptmr(args).to(dev).train()
    ksp = torch.randn(1, C, H, W, 2, device=dev)
    mask = torch.zeros(1, 1, 1, W, 1, dtype=torch.uint8, device=dev)
    mask[:, :, :, ::4, :] = 1                     # acc4 언더샘플
    mask[:, :, :, W // 2 - 15 : W // 2 + 15, :] = 1  # ACS
    tgt = torch.randn(1, 384, 384, device=dev)
    if dev == "cuda":
        torch.cuda.reset_peak_memory_stats()
    try:
        out = model(ksp, mask)
        loss = torch.nn.functional.l1_loss(out, tgt)
        loss.backward()
        if dev == "cuda":
            peak = torch.cuda.max_memory_allocated() / 1024**3
            print(f"OK   {tag} peak={peak:.2f}GB")
        else:
            print(f"OK (cpu) {tag}")
    except torch.cuda.OutOfMemoryError:
        print(f"OOM  {tag}")
        torch.cuda.empty_cache()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-cascades", type=int, default=10)
    ap.add_argument("--n-feat0", type=int, default=48)
    ap.add_argument("--sens-n-feat0", type=int, default=24)
    ap.add_argument("--n-history", type=int, default=3)
    ap.add_argument("--no-use-checkpoint", dest="use_checkpoint", action="store_false")
    # 코일 압축을 안 쓰기로 했으므로 최악 shape(readout-crop 384 × 최대폭 480 × 물리코일 15)를
    # 그대로 재볼 수 있어야 한다. 기본값은 기존 동작(H=640, W=368) 유지.
    ap.add_argument("--crop-h", type=int, default=640, help="readout(높이). --readout-crop 학습이면 384")
    ap.add_argument("--width", type=int, default=368, help="PE 폭. 학습셋 최대는 480(파일 1개)")
    ap.add_argument("--coils", type=int, default=15, help="코일 수. 압축 없이 전부 쓰면 15")
    ap.set_defaults(use_checkpoint=True)
    a = ap.parse_args()
    probe(a.num_cascades, a.n_feat0, a.sens_n_feat0, a.n_history, a.use_checkpoint,
          H=a.crop_h, W=a.width, C=a.coils)
