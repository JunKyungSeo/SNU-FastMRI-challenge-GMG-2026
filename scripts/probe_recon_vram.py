#!/usr/bin/env python3
"""recon_eval 추론 경로의 VRAM peak 측정 — 약한 GPU(4GB 등)에서 채점이 되는지 판정.

recon_eval.py 가 실제로 부르는 심볼(load_model → prep_volume → recon_slice)을 **그대로** 써서
슬라이스 몇 장을 복원하고 peak CUDA 메모리를 찍는다. 학습이 아니라 추론이라 no_grad 하에서
돌고, backward 그래프를 안 쌓으므로 학습 peak(3.35GB)보다 훨씬 낮게 나오는 게 정상이다.

가장 큰 레버는 cascade 단위 gradient checkpointing이다. test_part.load_model 이 추론에선
이걸 꺼두는데(`model.use_checkpoint = False`), 켜면 peak 가 내려가는 대신 느려진다.
추론시간을 신경 안 쓰는 환경이라면 사실상 공짜 레버라 두 경우를 다 재본다.

    python scripts/probe_recon_vram.py -n promptmr_moe
    python scripts/probe_recon_vram.py -n promptmr_moe --use-checkpoint

기본 타깃은 이 데이터셋의 **최악 폭 슬라이스**(W=480). 대부분 슬라이스는 W=368/372라 더 작다.
"""
import argparse
import os
import sys
from pathlib import Path

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(1, os.path.join(_ROOT, "utils", "model"))  # utils/model/fastmri 를 top-level로

import h5py     # noqa: E402
import torch    # noqa: E402

from utils.learning.test_part import load_model, prep_volume, recon_slice   # noqa: E402

DEFAULT_KSPACE = "/workspace/Data/train/kspace/knee_acc4_46.h5"   # W=480, 이 데이터셋 최악 폭


def parse():
    p = argparse.ArgumentParser(description="recon_eval 추론 VRAM peak 측정")
    p.add_argument("-n", "--net-name", required=True, help="채점할 net_name (../result/<net>/checkpoints/best_model.pt)")
    p.add_argument("-p", "--kspace", type=Path, default=Path(DEFAULT_KSPACE), help="측정에 쓸 k-space h5")
    p.add_argument("--slices", type=int, default=3, help="복원할 슬라이스 수(peak는 1장에서 거의 결정됨)")
    p.add_argument("--use-checkpoint", action="store_true",
                   help="추론에서도 cascade checkpointing 을 켠다(load_model 기본은 off). 느려지는 대신 peak↓")
    p.add_argument("--result-root", default="../result")
    # load_model 이 varnet 분기에서 참조하는 값들(promptmr 체크포인트면 안 쓰임)
    p.add_argument("--cascade", type=int, default=1)
    p.add_argument("--chans", type=int, default=9)
    p.add_argument("--sens_chans", type=int, default=4)
    return p.parse_args()


def _set_checkpoint(model, flag):
    """MoEWrapper면 전문가 각각에, 단일 모델이면 자기 자신에 플래그를 심는다."""
    targets = [model]
    for name in ("expert4", "expert8"):
        e = getattr(model, name, None)
        if e is not None:
            targets.append(e)
    for t in targets:
        t.use_checkpoint = flag
    return len(targets)


def main():
    args = parse()
    args.exp_dir = Path(args.result_root) / args.net_name / "checkpoints"
    if not (args.exp_dir / "best_model.pt").exists():
        sys.exit(f"[STOP] 체크포인트 없음: {args.exp_dir/'best_model.pt'}")
    if not args.kspace.exists():
        sys.exit(f"[STOP] k-space 파일 없음: {args.kspace}")

    cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if cuda else "cpu")
    if not cuda:
        print("[warn] CUDA 없음 → CPU로 돈다. VRAM 측정은 의미 없고 '동작 여부'만 확인된다.")

    with h5py.File(args.kspace, "r") as hf:
        shape = hf["kspace"].shape
    print(f"[probe] net={args.net_name}  use_checkpoint={args.use_checkpoint}")
    print(f"[probe] 입력 {args.kspace.name}  shape={shape}  (W={shape[-1]})")

    if cuda:
        torch.cuda.reset_peak_memory_stats()
    model = load_model(args, device)
    n_set = _set_checkpoint(model, args.use_checkpoint)
    kind = "MoE(전문가 2개)" if hasattr(model, "expert4") else "단일 모델"
    n_param = sum(p.numel() for p in model.parameters())
    w_mb = torch.cuda.memory_allocated() / 1024**2 if cuda else n_param * 4 / 1024**2
    print(f"[probe] {kind} · param {n_param/1e6:.1f}M · 로드 직후 {w_mb:.0f} MB "
          f"(checkpoint 플래그 {n_set}곳에 적용)")

    ctx = prep_volume(None, args.kspace, device)
    n = min(args.slices, ctx["num_slices"])
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    try:
        with torch.no_grad():
            for s in range(n):
                recon_slice(model, ctx, s)
        if cuda:
            torch.cuda.synchronize()
    except torch.cuda.OutOfMemoryError as e:                      # noqa: F821
        print(f"[RESULT] ❌ OOM — 이 설정으로는 안 된다.\n         {str(e)[:200]}")
        sys.exit(1)
    except RuntimeError as e:
        if "out of memory" not in str(e).lower():
            raise
        print(f"[RESULT] ❌ OOM — 이 설정으로는 안 된다.\n         {str(e)[:200]}")
        sys.exit(1)

    if cuda:
        peak_a = torch.cuda.max_memory_allocated() / 1024**3
        peak_r = torch.cuda.max_memory_reserved() / 1024**3
        print(f"[RESULT] ✅ {n} slice 복원 성공")
        print(f"[RESULT] peak allocated = {peak_a:.2f} GB / reserved = {peak_r:.2f} GB "
              f"(가중치 {w_mb/1024:.2f} GB 포함)")
        for limit in (4, 6, 8):
            ok = "들어감" if peak_r < limit * 0.92 else "위험"
            print(f"           {limit}GB GPU: {ok}  (reserved 기준, 여유 8% 가정)")
    else:
        print(f"[RESULT] ✅ CPU에서 {n} slice 복원 성공 (VRAM 무관)")


if __name__ == "__main__":
    main()
