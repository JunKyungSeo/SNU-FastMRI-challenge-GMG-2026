#!/usr/bin/env bash
# ============================================================
# GPU 작업 큐 슬롯 — PromptMR+ 실학습 4차 (promptmr_run4): full coil + nc10 + lr 2e-4.
#   이름 주의: run1=shell12(압축8·nc4) / run2=shell14(전코일·nc10) / run3=shell16(압축8·nc4·lr고침)
#   → 이 슬롯은 **run4**. 기존 결과 폴더를 덮어쓰지 않는다.
#
# 목적: "run2의 구조(물리코일 15 전부 + nc10) + run3의 lr 수정(2e-4)" 조합.
#   run2는 구조는 맞았지만 lr 1e-6이라 미학습이었고, run3은 lr은 고쳤지만 구조가 run1(압축8·nc4)로
#   되돌아갔다. 둘의 장점을 합친 게 이 슬롯이다.
#
#   [왜 full coil 인가]
#     - coil compression을 강제했던 근거("8GB에 15코일은 nc=1조차 OOM")는 gradient checkpointing이
#       실제로 안 걸린 상태에서 잰 값이었다. strategy/promptmr_plus.md 「재검토(2026-07-20)」가
#       그 결론을 스스로 철회했고, commit 0f16f8f("coil 압축 제거 준비")가 cascade·coil 단위
#       checkpointing을 제대로 걸어 제약을 없앴다. run2가 이 config로 실제 완주했다.
#     - 점수 비대칭: 추론시간은 ~0.00n 단위 타이브레이커인데 SSIM_bbox는 점수의 50%다.
#       SVD 절단은 전역 에너지 기준이라 저SNR 병변이 버려지는 성분에 묻힐 수 있고(preprocessing.md:65),
#       이 리스크는 한 번도 측정된 적이 없다(validate()가 SSIM_bbox를 아예 계산 안 함).
#     - 시간 손해는 직관보다 작다. cascade 본체 denoiser는 sens_reduce로 코일이 합쳐진 뒤 돌아
#       코일 수와 무관하다. 비용 ≈ nc + 0.25*C → nc10에서 C 8→15는 12.0→13.75 (+15%)이지,
#       15/8=1.875배가 아니다. 게다가 압축의 CPU SVD(슬라이스당 6.7~9.1ms, 실측)는 채점 타이머
#       안에 있는데(test_part.py:61-62 ⊂ recon_eval.py:80-83) full coil은 이걸 아예 건너뛴다.
#
#   [lr] 2e-4 상수. PromptMR+ 원본·참고 repo가 lr 2e-4 + StepLR(step 11, gamma 0.3)인데,
#        step_size 11 > epoch 10 이라 이번 런에서 감쇠는 발동하지 않는다 = 상수 2e-4와 동일하다.
#        따라서 --lr-step-size 배선(아직 train.py에 없음)을 기다릴 필요가 없다.
#        ⚠️ 나중에 resume으로 11 epoch을 넘길 거라면 그때는 StepLR 배선을 먼저 끝내야 한다.
#
#   [run3과의 비교 주의] run3 대비 coil(8→15)·cascade(4→10) 두 축이 함께 바뀐다. 즉 이 슬롯은
#        "최고 점수 후보"이지 "coil 축 단독 격리 실험"이 아니다.
#        coil만 격리하고 싶으면 --num-cascades 를 4로 낮춰서 돌릴 것.
#
#   [비용] 계산량이 run3(nc4·C8=6.0)의 약 2.3배(nc10·C15=13.75). 같은 10 epoch이면 벽시계 시간도
#        대략 그만큼 늘어난다고 보고 큐 일정을 잡을 것.
#
#   ⚠️ 미실측: full coil·nc10·최악폭(W=480)의 CUDA peak가 repo에 기록된 적이 없다. run2가 완주했다는
#      보고는 있으나 수치가 없다. → 반드시 SMOKE=1 을 먼저 돌려 넓은폭 슬라이스를 관통시킬 것.
#      학습 로그의 '[mem] epoch N peak allocated' 줄을 이번엔 꼭 training_anal/ 에 남긴다.
#
# 실행:  SMOKE=1 bash shell_scripting/shell18.sh    (120 iter, 넓은폭 관통 OOM 확인 — 먼저 이걸)
#        bash shell_scripting/shell18.sh            (본 학습 10 epoch + 복원·채점)
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
    NET="promptmr_run5_smoke"; TAG="pmr_run5_smoke"; EXTRA="--max-iters 120"; EPOCHS=1
else
    NET="promptmr_run5";       TAG="pmr_run5";       EXTRA="";                EPOCHS=10
fi

echo "=========================================================="
echo "[START] $SLOT   $(date '+%F %T')   SMOKE=$SMOKE  NET=$NET"
echo "[CWD  ] $(pwd)"
echo "[LOG  ] $LOG"
echo "[CONF ] full coil(압축 없음) · nc10 · nf48 · sens24 · lr 2e-4 상수 · ${EPOCHS} epoch"
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

# ==========================================================
#  ↓↓↓  1) PromptMR+ 학습 (full coil, scratch)  ↓↓↓
# ==========================================================
set -x
python train.py --model promptmr --readout-crop \
    --num-cascades 10 --n-feat0 48 --sens-n-feat0 24 --n-history 3 \
    -b 1 -l 2e-4 -e "$EPOCHS" -r 100 --num-workers 4 --use-checkpoint  \
    --fg-loss --area-weight --grad-clip 0.1 --accum-steps 2 $EXTRA \
    -n "$NET" --run-tag "$TAG" \
    -t /workspace/Data/train/ -v /workspace/Data/val/
{ TRAIN_STATUS=$?; set +x; } 2>/dev/null
echo "[TRAIN] $NET 종료   exit=$TRAIN_STATUS"
echo "[MEM  ] 위 로그의 '[mem] epoch N peak allocated' 줄을 training_anal/ 에 기록할 것 —"
echo "        full coil·nc10 의 실측 peak는 이 repo에 아직 남은 적이 없다."

if [ "$SMOKE" = "1" ]; then
    echo "=========================================================="
    echo "[SMOKE] 채점 생략. 확인할 것 두 가지:"
    echo "        (1) OOM 없이 exit 0 인가 — 120 iter 안에 넓은폭(W=480) 슬라이스를 지난다."
    echo "        (2) iter 0 대비 Loss 가 내려가는가."
    echo "        둘 다 OK면  bash shell_scripting/shell18.sh  로 본 학습(10 epoch)."
    echo "=========================================================="
    exit "$TRAIN_STATUS"
fi

# ==========================================================
#  ↓↓↓  2) 복원 + 채점 (recon_eval: reconstruct & score 동시)  ↓↓↓
#     복원 산출물: ../result/$NET/reconstructions_leaderboard/acc{4,8}
# ==========================================================
if [ "$TRAIN_STATUS" -eq 0 ]; then
    mkdir -p "../result/$NET"
    echo "[RECON] $NET 복원·채점 (SSIM_full + SSIM_bbox + 추론시간)..."
    set -x
    python recon_eval.py -n "$NET" -p /root/Data/leaderboard 2>&1 \
        | tee "../result/$NET/recon_eval_run5.txt"
    { STATUS=$?; set +x; } 2>/dev/null
else
    echo "[RECON] 학습이 exit=$TRAIN_STATUS 로 끝나 복원·채점 건너뜀."
    STATUS=$TRAIN_STATUS
fi

echo "=========================================================="
echo "[DONE ] $SLOT   $(date '+%F %T')   train=$TRAIN_STATUS recon/exit=$STATUS"
echo "[저장] 점수: ../result/$NET/recon_eval_run4.txt"
echo "       복원: ../result/$NET/reconstructions_leaderboard/acc{4,8}"
echo "[비교] run3(../result/promptmr_run3/recon_eval_run3.txt)와 3축 전부 비교할 것."
echo "       SSIM_bbox 차이가 핵심이다 — 추론시간은 +15% 예상이나 점수 기여는 ~0.00n 단위다."
echo "=========================================================="
exit "$STATUS"
