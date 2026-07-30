#!/usr/bin/env bash
# ============================================================
# GPU 작업 큐 슬롯 — PromptMR+ 실학습 3차 (promptmr_run3).  ※ shell12(run1)·shell14(run2)는 보존.
#   이름 주의: promptmr_run1=shell12(코일압축8·nc4), promptmr_run2=shell14(15코일 전부·nc10) 가 이미 점유.
#   → 이 슬롯은 **run3**. 기존 결과 폴더를 덮어쓰지 않는다.
#
# 목적: run1·run2 공통의 리더보드 SSIM 저조 원인(= 학습률 1e-6)을 고쳐 재학습.
#   run1/run2는 둘 다 -l 1e-6 이라 **아키텍처 대조(코일압축 vs 전코일) 자체가 무의미했다**
#   — 두 팔 모두 미학습 상태였기 때문. lr을 고친 뒤에 대조해야 의미가 생긴다.
#   [원인] run1은 -l 1e-6 상수. PromptMR+ 원본(hellopipu/PromptMR-plus)과 참고 repo
#          (sMartify 2025_final) 모두 **lr 2e-4 + StepLR** 이다 → 우리는 200배 낮았다.
#          정량: 5 epoch × 4651 iter ÷ accum2 ≈ 11.6k opt-step, Adam 스텝≈lr 이므로
#          가중치 총이동 상한 = 11.6k×1e-6 = 0.0116 < 초기화 RMS(~0.017~0.028).
#          즉 모델이 **랜덤 초기화에서 거의 못 움직인 채** 채점됐다.
#          (PromptMR+는 감도맵도 학습으로 얻으므로 미학습 시 zero-filled보다도 나쁘다.)
#   [수정] -l 2e-4 + StepLR(step 11, gamma 0.3) + -e 10  ← PromptMR+ 자체에서 검증된 레시피.
#          ※ SDUM 시절 2.4e-4는 다른 아키텍처 값이라 근거로 쓰지 않음.
#   [epoch 10 / step_size 11 유지 이유] 참고 repo는 epoch 11에서 처음 감쇠 → 그들의 epoch 0~10은
#          상수 2e-4다. 즉 -e 10 + step 11 이면 **참고 궤적의 첫 10 epoch을 그대로 재현**한다.
#          아직 overfitting이 관측된 적 없어(=underfit 구간) LR을 일찍 낮출 이유도 없다.
#          나중에 resume으로 11 epoch을 넘기면 감쇠가 설계대로 자동으로 걸린다.
#          ⚠️ 따라서 이번 런에서 StepLR은 실제로 발동하지 않는다(= 상수 2e-4). 정상이다.
#
# 메모리 config는 run1과 동일(8GB 실측 근거, strategy/promptmr_plus.md 하단):
#   coil-compress(15→8) + readout-crop + nc4 → 최악폭 W=480 실peak 6.06GB, 여유 1.94GB.
#   ⚠️ readout-crop/coil-compress는 recon_slice(test_part.py)에 배선되어 학습·채점이 자동 정합.
#
# ⚠️ 선행 조건: train.py 에 --lr-step-size / --lr-gamma 인자가 있어야 한다(아직 없으면 아래 가드가 막는다).
#    StepLR은 **epoch 단위**로 step 해야 한다(optimizer-step 단위로 물리면 11 step만에 LR이 꺾여 붕괴).
#
# 실행:  bash shell16.sh              (본 학습, GPU 비었을 때)
#        SMOKE=1 bash shell16.sh      (200 iter 스모크 — loss가 실제로 내려가는지 먼저 확인, 채점 생략)
# ============================================================
set -uo pipefail
export PYTHONUNBUFFERED=1

# --- repo 루트로 이동 ---
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.." || exit 1

# --- 로그 파일 ---
SLOT="$(basename "$0" .sh)"
TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="shell_scripting/logs"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/${SLOT}_${TS}.log"
exec > >(tee -a "$LOG") 2>&1

SMOKE="${SMOKE:-0}"
if [ "$SMOKE" = "1" ]; then
    NET="promptmr_run3_smoke"; TAG="pmr_run3_smoke"; EXTRA="--max-iters 200"; EPOCHS=1
else
    NET="promptmr_run3";       TAG="pmr_run3";       EXTRA="";                EPOCHS=10
fi

echo "=========================================================="
echo "[START] $SLOT   $(date '+%F %T')   SMOKE=$SMOKE  NET=$NET"
echo "[CWD  ] $(pwd)"
echo "[LOG  ] $LOG"
echo "=========================================================="

# --- 가드: GPU 비었는지 (used < 500MiB) ---
USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
echo "[GPU ] used=${USED}MiB"
if [ -n "$USED" ] && [ "$USED" -ge 500 ]; then
    echo "[STOP] GPU가 사용 중(${USED}MiB)이다. 실학습은 GPU가 빈 상태에서 실행할 것."
    exit 2
fi

# --- cv2 보장 (fg-loss·recon_eval metrics가 cv2 import) ---
python -c "import cv2" 2>/dev/null || { echo "[STOP] cv2 import 실패 — bash /root/restore_env.sh 로 환경 복구 후 실행할 것."; exit 2; }

# --- 가드: StepLR 인자가 train.py 에 배선되어 있는지 (없으면 fail-fast) ---
if ! python train.py --help 2>/dev/null | grep -q -- '--lr-step-size'; then
    echo "[STOP] train.py 에 --lr-step-size/--lr-gamma 가 없다. StepLR 배선을 먼저 끝낼 것."
    echo "       (StepLR은 반드시 epoch 단위 step — optimizer-step 단위면 LR이 즉시 붕괴)"
    exit 2
fi

# ==========================================================
#  ↓↓↓  1) PromptMR+ 재학습 (scratch, lr 2e-4 + StepLR)  ↓↓↓
# ==========================================================
set -x
python train.py --model promptmr --readout-crop --coil-compress --n-virtual-coils 8 \
    --num-cascades 4 --n-feat0 48 --sens-n-feat0 24 --n-history 3 --use-checkpoint \
    -b 1 -l 2e-4 --lr-step-size 11 --lr-gamma 0.3 -e "$EPOCHS" -r 100 --num-workers 4 \
    --fg-loss --area-weight --grad-clip 0.1 --accum-steps 2 $EXTRA \
    -n "$NET" --run-tag "$TAG" \
    -t /root/Data/train/ -v /root/Data/val/
{ TRAIN_STATUS=$?; set +x; } 2>/dev/null
echo "[TRAIN] $NET 종료   exit=$TRAIN_STATUS"

if [ "$SMOKE" = "1" ]; then
    echo "=========================================================="
    echo "[SMOKE] 채점 생략. 로그에서 iter 0 대비 Loss 가 실제로 내려갔는지 확인할 것."
    echo "        내려가면  bash shell_scripting/shell16.sh  로 본 학습(10 epoch) 진행."
    echo "        안 내려가면 중단하고 보고 — lr 외 다른 원인이 남아 있다는 뜻."
    echo "=========================================================="
    exit "$TRAIN_STATUS"
fi

# ==========================================================
#  ↓↓↓  2) recon_eval 채점 (학습 성공 시에만)  ↓↓↓
# ==========================================================
if [ "$TRAIN_STATUS" -eq 0 ]; then
    echo "[RECON] $NET 채점 (recon_eval → leaderboard SSIM_full+bbox+시간)..."
    set -x
    python recon_eval.py -n "$NET" -p /root/Data/leaderboard 2>&1 \
        | tee "../result/$NET/recon_eval_run3.txt"
    { STATUS=$?; set +x; } 2>/dev/null
else
    echo "[RECON] 학습이 exit=$TRAIN_STATUS 로 끝나 채점 건너뜀."
    STATUS=$TRAIN_STATUS
fi

echo "=========================================================="
echo "[DONE ] $SLOT   $(date '+%F %T')   train=$TRAIN_STATUS recon/exit=$STATUS"
echo "[저장] 점수: ../result/$NET/recon_eval_run3.txt (+ 이 슬롯 로그)."
echo "[비교] run1(압축8·nc4)·run2(전코일·nc10) 둘 다 lr 1e-6(미학습) → run3 상승폭이 lr 수정 효과."
echo "=========================================================="
exit "$STATUS"
