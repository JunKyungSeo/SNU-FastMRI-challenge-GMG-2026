#!/usr/bin/env bash
# ============================================================
# GPU 작업 큐 슬롯 — PromptMR+ K-Fold fold 2 (promptmr_kfold_f2)          [shell20.sh]
#
# shell19(fold 1)의 후속. 분할·학습 config는 f1과 **동일**하게 두고(= f1↔f2를 같은 5-fold CV의
# 두 fold로 비교), 코드 쪽 버그 두 개를 고쳤다.
#
#   ① volume 식별자 충돌  (utils/data/load_data.py)
#      train/ 과 val/ 에는 이름이 같은 **서로 다른** 볼륨이 30쌍 있다
#      (train/…/knee_acc4_1.h5 = 23슬라이스, val/…/knee_acc4_1.h5 = 31슬라이스).
#      옛 구현은 파일명 set으로 분할해서 200 volume을 170개로 셌다 — shell19 로그의
#      "Train Volume 수: 136개 | Val Volume 수: 34개"는 volume 수가 아니라 **이름 수**였고,
#      실제로는 fold별 39~43 volume이었다. 더 나쁜 건 validate()가 reconstructions[fname]을
#      파일명으로 키잉해서 이름이 같은 두 볼륨의 슬라이스가 한 항목에서 서로 덮어썼다는 점이다
#      (--fg-loss가 켜져 있어 val loss 값 자체는 무사했지만, NewRecord마다 저장되는
#       reconstructions_val/ 은 train·val 슬라이스가 섞인 볼륨이었다).
#      → volume_id("train__knee_acc4_1.h5")로 root를 붙여 분리. 이제 로그가 실제 volume·slice
#        수를 찍는다. 파티션 자체는 f1과 비트 단위로 같다(5개 fold 전부 대조 확인).
#
#   ② 채점이 항상 죽던 문제
#      이 서버엔 /root/Data/leaderboard 가 없다. f1은 21시간 학습을 마치고 recon_eval.py:120의
#      AssertionError로 끝나 점수를 한 줄도 못 남겼다(promptmr_run4도 같은 이유로 실패).
#      → hold-out fold를 leaderboard 형식으로 링크해서 recon_eval.py(고정·수정 금지)를 그대로
#        돌린다. fold2 hold-out = 40 volume · 1101 slice · 병변 566박스, 전부 학습에 안 쓴
#        데이터라 정직한 SSIM_full·SSIM_bbox·추론시간이 나온다.
#      ⚠️ leaderboard 점수와 같은 수는 아니다(분포·박스 수가 다름). fold끼리·run끼리 비교용.
#
#   [fold2 구성]  train 160 vol (4341 slice)  /  val 40 vol (acc4 19 · acc8 21, 1101 slice)
#   [config]      f1과 동일: full coil · nc10 · nf48 · sens24 · n-history3 · lr 2e-4 상수 ·
#                 fg-loss + area-weight · grad-clip 0.1 · accum 2 · checkpointing on · 10 epoch
#                 → 바꾸지 말 것. 바꾸면 f1과 비교가 안 된다(그 실험은 별도 슬롯에서).
#
# 실행:
#   SMOKE=1 bash shell_scripting/shell20.sh   # 120 iter — OOM 없이 exit 0 · 손실 하강 확인 (먼저 이것)
#   bash shell_scripting/shell20.sh           # 본 학습 10 epoch + hold-out fold 채점 (f1 기준 ~21시간)
#   FOLD=3 bash shell_scripting/shell20.sh    # 다른 fold를 돌릴 때
#   FORCE=1 ...                               # 기존 결과 폴더를 덮어쓸 때만 (기본은 거부)
# ============================================================
set -uo pipefail
export PYTHONUNBUFFERED=1

# --- repo 루트로 이동 ---
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.." || exit 1

# --- K-Fold 설정 (train.py --kfold-seed 기본값 430과 맞출 것) ---
FOLD="${FOLD:-2}"
K_FOLD="${K_FOLD:-5}"
KFOLD_SEED="${KFOLD_SEED:-430}"

# --- 로그 파일 ---
SLOT="$(basename "$0" .sh)"
TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="shell_scripting/logs"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/${SLOT}_f${FOLD}_${TS}.log"
exec > >(tee -a "$LOG") 2>&1

SMOKE="${SMOKE:-0}"
if [ "$SMOKE" = "1" ]; then
    NET="promptmr_kfold_f${FOLD}_smoke"; TAG="pmr_kf_f${FOLD}_smoke"; EXTRA="--max-iters 120"; EPOCHS=1
else
    NET="promptmr_kfold_f${FOLD}";       TAG="pmr_kf_f${FOLD}";       EXTRA="";                EPOCHS=10
fi
EVAL_DIR="../result/kfold_eval/f${FOLD}"

echo "=========================================================="
echo "[START] $SLOT (fold $FOLD / $K_FOLD)   $(date '+%F %T')   SMOKE=$SMOKE  NET=$NET"
echo "[CWD  ] $(pwd)"
echo "[LOG  ] $LOG"
echo "[CONF ] full coil · nc10 · nf48 · sens24 · lr 2e-4 · K=$K_FOLD · fold=$FOLD · seed=$KFOLD_SEED · ${EPOCHS} epoch"
echo "=========================================================="

# --- 가드: GPU 비었는지 (used < 500MiB) ---
USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
echo "[GPU  ] used=${USED}MiB"
if [ -n "$USED" ] && [ "$USED" -ge 500 ]; then
    echo "[STOP] GPU가 사용 중(${USED}MiB)이다. 실학습은 GPU가 빈 상태에서 실행할 것."
    exit 2
fi

# --- cv2 보장 (fg-loss·recon_eval metrics가 cv2 import) ---
python -c "import cv2" 2>/dev/null || { echo "[STOP] cv2 import 실패 — bash /root/restore_env.sh 로 환경 복구 후 실행할 것."; exit 2; }

# --- 가드: 기존 학습 결과를 말없이 덮어쓰지 않는다 (f1이 21시간짜리다) ---
if [ -e "../result/$NET/checkpoints/model.pt" ] && [ "${FORCE:-0}" != "1" ]; then
    echo "[STOP] ../result/$NET/checkpoints/model.pt 가 이미 있다 — 덮어쓰면 이전 학습 결과가 사라진다."
    echo "       이어서 학습하려면 train.py에 --resume 을 붙이고, 정말 새로 덮어쓸 거면 FORCE=1 로 실행할 것."
    exit 2
fi

# ==========================================================
#  ↓↓↓  0) hold-out fold 채점 세트 준비 (심볼릭 링크, GPU 미사용)  ↓↓↓
#     먼저 만들어 둔다 — 21시간 학습을 마친 뒤에야 채점이 안 된다는 걸 아는 일이 없도록.
# ==========================================================
set -x
python scripts/make_fold_eval_dir.py \
    --k-fold "$K_FOLD" --val-fold "$FOLD" --kfold-seed "$KFOLD_SEED" -o "$EVAL_DIR" \
    -t /workspace/Data/train/ -v /workspace/Data/val/
{ EVAL_READY=$?; set +x; } 2>/dev/null
[ "$EVAL_READY" -eq 0 ] || echo "[WARN] 채점 세트 생성 실패(exit=$EVAL_READY) — 학습은 진행하되 채점은 건너뛴다."

# ==========================================================
#  ↓↓↓  1) PromptMR+ K-Fold 학습 (fold $FOLD)  ↓↓↓
# ==========================================================
set -x
python train.py --model promptmr --readout-crop \
    --num-cascades 10 --n-feat0 48 --sens-n-feat0 24 --n-history 3 \
    -b 1 -l 2e-4 -e "$EPOCHS" -r 100 --num-workers 4 --use-checkpoint \
    --fg-loss --area-weight --grad-clip 0.1 --accum-steps 2 \
    --k-fold "$K_FOLD" --val-fold "$FOLD" --kfold-seed "$KFOLD_SEED" $EXTRA \
    -n "$NET" --run-tag "$TAG" \
    -t /workspace/Data/train/ -v /workspace/Data/val/
{ TRAIN_STATUS=$?; set +x; } 2>/dev/null
echo "[TRAIN] $NET (fold $FOLD) 종료   exit=$TRAIN_STATUS"

if [ "$SMOKE" = "1" ]; then
    echo "=========================================================="
    echo "[SMOKE] 채점 생략 (fold $FOLD). 확인할 것 세 가지:"
    echo "        (1) OOM 없이 exit 0 인가"
    echo "        (2) iter 0 대비 Loss 가 내려가는가"
    echo "        (3) '[K-Fold] train 160 vol (4341 slice) / val 40 vol (1101 slice)' 로 찍혔는가"
    echo "            — 예전처럼 '136개/34개'(이름 수)가 나오면 코드가 옛 버전이다."
    echo "        전부 OK면  bash shell_scripting/shell20.sh  로 본 학습."
    echo "=========================================================="
    exit "$TRAIN_STATUS"
fi

# ==========================================================
#  ↓↓↓  2) hold-out fold 복원 + 채점 (recon_eval, 고정 하네스)  ↓↓↓
# ==========================================================
STATUS=$TRAIN_STATUS
if [ "$TRAIN_STATUS" -ne 0 ]; then
    echo "[RECON] 학습이 exit=$TRAIN_STATUS 로 끝나 복원·채점 건너뜀."
elif [ ! -d "$EVAL_DIR/acc4" ]; then
    echo "[RECON] 채점 세트($EVAL_DIR)가 없어 건너뜀 — 아래를 직접 돌릴 것:"
    echo "        python scripts/make_fold_eval_dir.py --k-fold $K_FOLD --val-fold $FOLD -o $EVAL_DIR"
    echo "        python recon_eval.py -n $NET -p $EVAL_DIR"
else
    mkdir -p "../result/$NET"
    echo "[RECON] $NET — hold-out fold $FOLD 복원·채점 (SSIM_full + SSIM_bbox + 추론시간)..."
    set -x
    python recon_eval.py -n "$NET" -p "$EVAL_DIR" 2>&1 \
        | tee "../result/$NET/recon_eval_kfold_f${FOLD}.txt"
    { STATUS=$?; set +x; } 2>/dev/null
fi

# --- leaderboard 데이터가 배포되면 그것도 함께 채점 (지금 이 서버엔 없다) ---
for LB in /root/Data/leaderboard /workspace/Data/leaderboard; do
    if [ -d "$LB/acc4" ] && [ -d "$LB/acc8" ] && [ "$TRAIN_STATUS" -eq 0 ]; then
        echo "[RECON] leaderboard 발견($LB) — 공식 채점도 실행."
        set -x
        python recon_eval.py -n "$NET" -p "$LB" 2>&1 \
            | tee "../result/$NET/recon_eval_leaderboard.txt"
        { set +x; } 2>/dev/null
        break
    fi
done

echo "=========================================================="
echo "[DONE ] $SLOT (fold $FOLD)   $(date '+%F %T')   train=$TRAIN_STATUS recon/exit=$STATUS"
echo "[저장] 점수    : ../result/$NET/recon_eval_kfold_f${FOLD}.txt"
echo "       val loss: ../result/$NET/val_loss_log_${TAG}.npy"
echo "       가중치  : ../result/$NET/checkpoints/best_model.pt"
echo "[비교] f1(../result/promptmr_kfold_f1/val_loss_log_pmr_kf_f1.npy, 최종 ValLoss 0.09357)와"
echo "       같은 파티션·같은 config다 → val loss를 fold 간 편차로 읽어도 된다."
echo "       ⚠️ f1은 채점이 죽어 SSIM 점수가 없다. f1도 점수를 보려면:"
echo "          python scripts/make_fold_eval_dir.py --k-fold 5 --val-fold 1 -o ../result/kfold_eval/f1"
echo "          python recon_eval.py -n promptmr_kfold_f1 -p ../result/kfold_eval/f1"
echo "=========================================================="
exit "$STATUS"
