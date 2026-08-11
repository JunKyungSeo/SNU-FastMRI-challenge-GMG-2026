#!/usr/bin/env bash
# ============================================================
# 목적: 최신 acc8(_ext 있으면 그것) + acc4 를 병합해 채점            [shell31_acc8more_recon.sh]
#       acc8 을 더 돌릴 때마다 재병합해야 하므로 병합은 매번 다시 한다.
#
# 실행:  bash shell_scripting/shell31_acc8more_recon.sh
#        ACC8_NET=promptmrp_0801_acc8 bash shell_scripting/shell31_acc8more_recon.sh   # 원본으로
# ============================================================
set -uo pipefail
export PYTHONUNBUFFERED=1

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.." || exit 1

LEADERBOARD="${LEADERBOARD:-/root/Data/leaderboard}"
RUN="promptmrp_0801"
ACC4_NET="${ACC4_NET:-${RUN}_acc4}"
# _ext 가 있으면 그게 최신이다.
DEFAULT_ACC8="${RUN}_acc8"
[ -f "../result/${RUN}_acc8_ext/checkpoints/best_model.pt" ] && DEFAULT_ACC8="${RUN}_acc8_ext"
ACC8_NET="${ACC8_NET:-$DEFAULT_ACC8}"
OUT="${OUT:-${RUN}_moe_${ACC8_NET##*_}}"

for n in "$ACC4_NET" "$ACC8_NET"; do
    [ -f "../result/$n/checkpoints/best_model.pt" ] || { echo "[STOP] 체크포인트 없음: $n"; exit 2; }
done

SLOT="$(basename "$0" .sh)"
TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="shell_scripting/logs"; mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/${SLOT}_${TS}.log"
exec > >(tee -a "$LOG") 2>&1
OUT_DIR="shell_scripting/results"; mkdir -p "$OUT_DIR"
SCORE="$OUT_DIR/recon_eval_${OUT}_${TS}.txt"

echo "[START] $SLOT   $(date '+%F %T')   acc4=$ACC4_NET  acc8=$ACC8_NET  →  $OUT"
# 어느 시점의 전문가인지 로그에 남긴다 — _ext 가 원본보다 나쁠 수도 있다(best_val 이 새로 시작하므로).
python3 - "$ACC4_NET" "$ACC8_NET" "${RUN}_acc8" <<'EOF'
import sys, numpy as np
for n in dict.fromkeys(sys.argv[1:]):
    try:
        v = np.load(f"../result/{n}/val_loss_log_{n}.npy")
        print(f"[VAL  ] {n}: {len(v)}ep · 최저 {v[:,1].min():.5f}")
    except Exception as e:
        print(f"[VAL  ] {n}: 읽기 실패 ({e})")
EOF

python scripts/merge_moe.py --acc4-net "$ACC4_NET" --acc8-net "$ACC8_NET" --out-net "$OUT" || exit $?

set -x
python -u recon_eval.py -n "$OUT" -p "$LEADERBOARD" | tee "$SCORE"
{ ST=${PIPESTATUS[0]}; set +x; } 2>/dev/null

echo "[DONE ] $SLOT   $(date '+%F %T')   exit=$ST"
echo "[저장] 점수: $SCORE"
bash "$SCRIPT_DIR/notify.sh" "$OUT 채점 완료 (exit=$ST)" \
    "acc4=$ACC4_NET · acc8=$ACC8_NET · $(TZ=Asia/Seoul date '+%F %T') KST"$'\n\n'"$(grep -E 'SSIM_(full|bbox)|Recon Time' "$SCORE")"
exit "$ST"
