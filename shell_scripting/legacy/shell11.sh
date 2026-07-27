#!/usr/bin/env bash
# ============================================================
# GPU 작업 큐 슬롯 — PromptMR+ 스모크(crash-check + 8GB VRAM 적합). 테스트용·디스포저블.
# 목적: jkMRplus 백본(PromptMR+)이 실데이터에서
#       데이터로드 → 어댑터 forward → 손실 → backward → step 전 경로를
#       크래시·CUDA OOM 없이 도는지, 8GB-적합 config 로 확인.
#   ⚠️ config 근거(2026-07-19 메모리 실측): SPEC nc10·full-640 은 8GB OOM. 메모리는 coil/sens 지배.
#      k-space 폭이 가변(368~480)이고 readout-crop은 높이만 자르므로 최악폭 W=480에선 15 coil이
#      nc=1 조차 OOM. → coil compression(15→8, 99.999% 에너지 보존) + readout-crop + nc4 가 유일 해법.
#      최악폭 W=480 실peak 6.06GB(여유 1.94GB). 상세: strategy/promptmr_plus.md 하단.
#   ⚠️ --max-iters 120 인 이유: 좁은폭(W368) 몇 iter만 보면 최악폭 슬라이스를 놓쳐 헛통과(구 shell12 OOM
#      전례). seed 430 셔플에서 넓은폭 슬라이스가 iter 1~99 안에 나오므로 120 iter 로 그걸 반드시 거친다.
#   ⚠️ validate()는 --max-iters 캡이 없어 검증은 tiny_val(1볼륨) 로 축소(학습 config는 실전 그대로).
#   통과 기준: exit 0 + 로그에 TrainLoss/ValLoss 요약. 산출물(promptmr_smoke)은 확인 후 삭제 가능.
#
# 실행:  bash shell11.sh   (GPU 비었을 때)
# ============================================================
set -uo pipefail
export PYTHONUNBUFFERED=1   # tee 파이프로 가도 python 출력을 실시간으로

# --- repo 루트로 이동 (train.py 는 repo 루트 기준으로 돈다) ---
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.." || exit 1

# --- 로그 파일 (이름에 timestamp) ---
SLOT="$(basename "$0" .sh)"
TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="shell_scripting/logs"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/${SLOT}_${TS}.log"
exec > >(tee -a "$LOG") 2>&1

echo "=========================================================="
echo "[START] $SLOT   $(date '+%F %T')"
echo "[CWD  ] $(pwd)"
echo "[LOG  ] $LOG"
echo "=========================================================="

# --- 가드: GPU 비었는지 (used < 500MiB) — 다른 작업과 충돌 방지 ---
USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
echo "[GPU ] used=${USED}MiB"
if [ -n "$USED" ] && [ "$USED" -ge 500 ]; then
    echo "[STOP] GPU가 사용 중(${USED}MiB)이다. 스모크는 GPU가 빈 상태에서 실행할 것."
    exit 2
fi

# --- cv2 보장 (fg-loss가 metrics(cv2) import, headless 서버) ---
python -c "import cv2" 2>/dev/null || { echo "[STOP] cv2 import 실패 — bash /root/restore_env.sh 로 환경 복구 후 실행할 것."; exit 2; }

# --- 검증 축소용 tiny val (1 볼륨만 심볼릭 링크) ---
TINY_VAL="../result/promptmr_smoke/tiny_val"
rm -rf "$TINY_VAL"; mkdir -p "$TINY_VAL/image" "$TINY_VAL/kspace"
VOL="$(ls /root/Data/val/image/ | head -1)"
ln -s "/root/Data/val/image/$VOL"  "$TINY_VAL/image/$VOL"
ln -s "/root/Data/val/kspace/$VOL" "$TINY_VAL/kspace/$VOL"
echo "[VAL ] tiny_val = $VOL (검증 1 볼륨으로 축소)"

# ==========================================================
#  ↓↓↓  PromptMR+ 스모크 (8GB-적합 config: nc4·nf48·readout-crop + --max-iters 3)  ↓↓↓
# ==========================================================
set -x
python train.py --model promptmr --readout-crop --coil-compress --n-virtual-coils 8 \
    --num-cascades 4 --n-feat0 48 --sens-n-feat0 24 --n-history 3 --use-checkpoint \
    -b 1 -l 1e-6 -e 1 --max-iters 120 -r 50 --num-workers 4 \
    --fg-loss --area-weight --grad-clip 0.1 --accum-steps 2 \
    -n promptmr_smoke --run-tag smoke \
    -t /root/Data/train/ -v "$TINY_VAL"
{ STATUS=$?; set +x; } 2>/dev/null

# --- tiny val 정리 ---
rm -rf "$TINY_VAL"

echo "=========================================================="
if [ "$STATUS" -eq 0 ]; then
    echo "[PASS] PromptMR+ 스모크 통과 — 크래시·OOM 없이 3-iter 학습 + 검증 완료."
else
    echo "[FAIL] 스모크 실패 exit=$STATUS — 로그 위쪽 에러 확인(OOM/shape/wiring)."
fi
echo "[DONE ] $SLOT   $(date '+%F %T')   exit=$STATUS"
echo "[산출물] ../result/promptmr_smoke (디스포저블 — 확인 후 삭제 가능)"
echo "=========================================================="
exit "$STATUS"
