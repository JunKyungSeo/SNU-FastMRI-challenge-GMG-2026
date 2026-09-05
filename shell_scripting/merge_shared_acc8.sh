#!/usr/bin/env bash
# 목적: shared 를 acc4 자리에, acc8 계열(기본 _acc8_tail)을 acc8 자리에 넣어 MoE 병합
# 실행:  bash shell_scripting/merge_shared_acc8.sh [out_net]
set -uo pipefail
cd "$(cd "$(dirname "$0")" && pwd)/.." || exit 1
source shell_scripting/_final_base.sh

OUT="${1:-${RUN}_moe_s4a8tail}"
A8="${2:-${RUN}_acc8_tail}"   # acc8 자리에 넣을 net (기본 tail 결과)
python scripts/merge_moe.py --acc4-net "${RUN}_shared" --acc8-net "$A8" --out-net "$OUT" || exit $?
echo "[done ] ../result/$OUT/checkpoints/best_model.pt"
echo "[채점 ] python recon_eval.py -n $OUT -p $DATA_ROOT/leaderboard   ← GPU 필요, 학습과 동시 실행 불가"
