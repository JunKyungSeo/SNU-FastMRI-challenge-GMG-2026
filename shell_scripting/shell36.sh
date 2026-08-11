#!/usr/bin/env bash
# ============================================================
# 목적: cascade 10→12 · n_history 3→11 의 VRAM·시간 비용 측정                    [shell36.sh]
#       두 레버를 따로 재서 어느 쪽이 8GB 를 먹는지 가른다. 학습 0, 4분.
#
# 실행:  bash shell_scripting/shell36.sh
#        WIDTH=480 bash shell_scripting/shell36.sh   # 최악 폭(학습셋 최대)
# ============================================================
set -uo pipefail
export PYTHONUNBUFFERED=1

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.." || exit 1

WIDTH="${WIDTH:-368}"
SLOT="$(basename "$0" .sh)"
LOG_DIR="shell_scripting/logs"; mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/${SLOT}_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1

# 학습이 GPU 를 잡고 있으면 남은 메모리만 보고 OOM 이 나 측정이 무의미해진다.
USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
[ -n "$USED" ] && [ "$USED" -ge 500 ] && { echo "[STOP] GPU 사용 중(${USED}MiB) — 비운 뒤 돌릴 것"; exit 2; }

echo "[START] $SLOT   $(date '+%F %T')   crop-h=384  width=$WIDTH  coils=15"
for cfg in "10:3" "12:3" "10:11" "12:11"; do
    NC="${cfg%%:*}"; NH="${cfg##*:}"
    echo "--- cascade=$NC  n_history=$NH ---"
    python scripts/probe_oom_promptmr.py \
        --num-cascades "$NC" --n-history "$NH" \
        --n-feat0 48 --sens-n-feat0 24 \
        --crop-h 384 --width "$WIDTH" --with-optimizer
done

echo "[DONE ] $SLOT   $(date '+%F %T')"
echo "[해석] peak 이 8GB 에 붙으면 그 config 는 못 쓴다. fwdbwd 비율 × 5.90h = 새 1 epoch 시간."
bash "$SCRIPT_DIR/notify.sh" "$SLOT 탐침 완료" "$(grep -E '^(OK|OOM)' "$LOG")"
