"""acc4·acc8 전문가 체크포인트 2개를 MoE 결합 체크포인트 1개로 병합.

Option C 흐름: 공유 사전학습 → acc별 미세조정(전문가 2개) → 이 스크립트로 병합 → recon_eval로 채점.
결과물은 test_part.load_model이 인식하는 {'moe':True, 'experts':{...}, 'router':{...}} 형식.
채점 하네스(recon_eval.py)는 net_name 하나만 받으므로, 병합본을 하나의 net_name 아래에 둔다.

사용:
  python utils/model/merge_moe.py --acc4-net promptmr_moe_acc4 --acc8-net promptmr_moe_acc8 \
      --out-net promptmr_moe [--threshold 0.1875] [--result-root ../result]
그 뒤:
  python recon_eval.py -n promptmr_moe -p /root/Data/leaderboard
"""
import argparse
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, os.getcwd())  # repo 루트에서 실행 전제(다른 scripts와 동일 관례)
from utils.model.moe import DEFAULT_ACC_THRESHOLD

# 두 전문가가 반드시 일치해야 하는 args 키(전처리·아키텍처). recon_slice는 래퍼 하나의
# 전처리 속성만 쓰므로, 둘이 다르면 한쪽 전문가가 학습분포와 어긋난 입력을 받게 된다.
MUST_MATCH = ["model", "readout_crop", "crop_h", "coil_compress", "n_virtual_coils",
              "num_cascades", "n_feat0", "sens_n_feat0", "n_history"]


def _load_best(result_root, net):
    path = Path(result_root) / net / "checkpoints" / "best_model.pt"
    if not path.exists():
        raise FileNotFoundError(f"체크포인트 없음: {path}")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if "model" not in ckpt or "args" not in ckpt:
        raise ValueError(f"{path} 에 'model'/'args' 키가 없습니다(정상 학습 체크포인트가 아님).")
    return ckpt


def main():
    p = argparse.ArgumentParser(description="acc4/acc8 전문가 → MoE 결합 체크포인트 병합")
    p.add_argument("--acc4-net", required=True, help="acc4 전문가 net_name")
    p.add_argument("--acc8-net", required=True, help="acc8 전문가 net_name")
    p.add_argument("--out-net", required=True, help="병합본 net_name(채점 시 -n 으로 사용)")
    p.add_argument("--result-root", default="../result", help="result 루트(기본 ../result)")
    p.add_argument("--threshold", type=float, default=DEFAULT_ACC_THRESHOLD,
                   help=f"라우터 임계값 frac(기본 {DEFAULT_ACC_THRESHOLD}). frac<임계 → acc8")
    args = p.parse_args()

    c4 = _load_best(args.result_root, args.acc4_net)
    c8 = _load_best(args.result_root, args.acc8_net)
    a4, a8 = c4["args"], c8["args"]

    # 전처리·아키텍처 정합 검증 — 어긋나면 라우터가 잘못된 입력을 전문가에 줄 수 있어 fail-fast.
    mismatch = [k for k in MUST_MATCH if getattr(a4, k, None) != getattr(a8, k, None)]
    if mismatch:
        raise ValueError(
            "acc4/acc8 전문가의 config가 다릅니다 → 병합 불가(라우터가 전처리 하나만 공유).\n"
            + "\n".join(f"  {k}: acc4={getattr(a4, k, None)!r}  acc8={getattr(a8, k, None)!r}"
                        for k in mismatch))

    out = {
        "moe": True,
        "experts": {
            "acc4": {"args": a4, "model": c4["model"]},
            "acc8": {"args": a8, "model": c8["model"]},
        },
        "router": {"threshold": args.threshold},
        # 참고 메타(추론엔 불필요, 기록용).
        "source_nets": {"acc4": args.acc4_net, "acc8": args.acc8_net},
    }

    out_dir = Path(args.result_root) / args.out_net / "checkpoints"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "best_model.pt"
    torch.save(out, out_path)
    print(f"[merge] 저장: {out_path}")
    print(f"[merge] acc4={args.acc4_net}  acc8={args.acc8_net}  threshold={args.threshold}")
    print(f"[merge] 채점:  python recon_eval.py -n {args.out_net} -p /root/Data/leaderboard")


if __name__ == "__main__":
    main()
