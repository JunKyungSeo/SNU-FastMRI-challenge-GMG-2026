#!/usr/bin/env bash
# ============================================================
# GPU 작업 큐 슬롯 — PromptMR+ K-Fold fold 2 (promptmr_kfold_f2)          [shell21.sh]
#
#   ⚠️ 슬롯 번호 주의: shell20.sh 는 팀원의 MoE 파이프라인이다. 이 슬롯(단일 모델 K-Fold)과는
#      별개 실험이니 헷갈리지 말 것. 두 슬롯은 서로 독립이고 같이 돌리면 안 된다(GPU 1장).
#
# shell19(fold 1)의 후속. 학습 config는 f1과 동일하되, 코드 쪽 버그 두 개를 고쳤고
# **분할 방식이 f1과 달라졌다**(아래 ③).
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
#        수를 찍는다.
#
#   ② 분할 단위 = volume  (jkMRplus 병합 시 팀원 방식 채택)
#      이름이 같은 두 볼륨은 서로 남남인데 옛 방식은 둘을 묶어 같은 fold로 보냈다. 이제 각각
#      독립 단위라 fold마다 정확히 40개씩 떨어진다.
#      ⚠️ **f1(promptmr_kfold_f1)과는 다른 파티션이다.** f1의 val 43개 중 이 fold2와 겹치는
#         건 일부뿐이라, f1·f2를 "한 5-fold CV의 두 fold"로 묶어 평균 내면 안 된다.
#         각각 독립적인 hold-out 성적으로 읽을 것.
#
#   [fold2 구성]  train 160 vol (4343 slice)  /  val 40 vol (acc4 22 · acc8 18, 1099 slice)
#                 ⚠️ 교체 전(비층화) 기준이다. 층화 이후엔 acc 구성도 슬라이스 수도 달라진다
#                    (층화는 fold마다 acc4/acc8 비율을 같게 맞춘다).
#   [config]      f1과 동일: full coil · nc10 · nf48 · sens24 · n-history3 · lr 2e-4 상수 ·
#                 fg-loss + area-weight · grad-clip 0.1 · accum 2 · checkpointing on · 10 epoch
#
#   [범위]  이 슬롯은 **학습 + hold-out 검증까지만** 한다. recon_eval 채점은 leaderboard
#           데이터가 있는 다른 환경에서 하므로 여기선 안 돌린다(GPU 시간·디스크 절약).
#           산출물은 val_loss_log(.npy)와 checkpoints/best_model.pt 두 개다.
#
#   ⚠️ (2026-07-26 갱신) kfold_split이 acc 층화로 교체되어 --acc 4/8 을 줘도 fold 경계가
#      안 바뀐다(MoE 다단계 파이프라인 안전). 단, 이 슬롯으로 이미 완주한 f2 런은
#      **교체 전(비층화) 파티션** 산출물이라, 층화 이후 새로 도는 런과 val 숫자를
#      나란히 비교하면 안 된다 — load_data.kfold_split docstring 참고.
#
# 실행:
#   SMOKE=1 bash shell_scripting/shell21.sh   # 120 iter — OOM 없이 exit 0 · 손실 하강 확인 (먼저 이것)
#   bash shell_scripting/shell21.sh           # 본 학습 10 epoch (f1 기준 ~21시간)
#   FOLD=3 bash shell_scripting/shell21.sh    # 다른 fold를 돌릴 때
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

# --- cv2 보장 (--fg-loss 의 foreground mask가 cv2 import) ---
python -c "import cv2" 2>/dev/null || { echo "[STOP] cv2 import 실패 — bash /root/restore_env.sh 로 환경 복구 후 실행할 것."; exit 2; }

# --- 가드: 기존 학습 결과를 말없이 덮어쓰지 않는다 (f1이 21시간짜리다) ---
#if [ -e "../result/$NET/checkpoints/model.pt" ] && [ "${FORCE:-0}" != "1" ]; then
#    echo "[STOP] ../result/$NET/checkpoints/model.pt 가 이미 있다 — 덮어쓰면 이전 학습 ##결과가 사라진다."
#    echo "       이어서 학습하려면 train.py에 --resume 을 붙이고, 정말 새로 덮어쓸 거면 ##FORCE=1 로 실행할 것."
#    exit 2
#fi

# ==========================================================
#  ↓↓↓  PromptMR+ K-Fold 학습 (fold $FOLD)  ↓↓↓
#     이 슬롯은 학습 + hold-out 검증까지만 한다. 채점(recon_eval)은 leaderboard 데이터가
#     있는 다른 환경에서 하므로 여기서 GPU를 더 쓰지 않는다.
# ==========================================================
set -x
python train.py --model promptmr --readout-crop \
    --num-cascades 10 --n-feat0 48 --sens-n-feat0 24 --n-history 3 \
    -b 1 -l 2e-4 -e "$EPOCHS" -r 100 --num-workers 4 --use-checkpoint \
    --fg-loss --area-weight --grad-clip 0.1 --accum-steps 2 \
    --k-fold "$K_FOLD" --val-fold "$FOLD" --kfold-seed "$KFOLD_SEED" $EXTRA \
    -n "$NET" --run-tag "$TAG" --resume \
    -t /workspace/Data/train/ -v /workspace/Data/val/
{ TRAIN_STATUS=$?; set +x; } 2>/dev/null
echo "[TRAIN] $NET (fold $FOLD) 종료   exit=$TRAIN_STATUS"

if [ "$SMOKE" = "1" ]; then
    echo "=========================================================="
    echo "[SMOKE] 확인할 것 세 가지 (fold $FOLD):"
    echo "        (1) OOM 없이 exit 0 인가"
    echo "        (2) iter 0 대비 Loss 가 내려가는가"
    echo "        (3) fold2면 '[K-Fold] train 160 vol · 4343 slice / val 40 vol · 1099 slice' 로 찍혔는가"
    echo "            — 예전처럼 '136개/34개'(이름 수)가 나오면 코드가 옛 버전이다."
    echo "        전부 OK면  bash shell_scripting/shell21.sh  로 본 학습."
    echo "=========================================================="
fi

echo "=========================================================="
echo "[DONE ] $SLOT (fold $FOLD)   $(date '+%F %T')   train=$TRAIN_STATUS"
echo "[저장] val loss: ../result/$NET/val_loss_log_${TAG}.npy"
echo "       가중치  : ../result/$NET/checkpoints/best_model.pt   ← 채점은 이걸 다른 환경으로 옮겨서"
echo "[비교] ⚠️ f1(promptmr_kfold_f1, 최종 ValLoss 0.09357)은 병합 전 '이름 그룹' 분할로 학습됐다."
echo "       지금 분할(volume 단위)과 hold-out이 달라 val loss를 나란히 비교하면 안 된다."
echo "       f2 이후 슬롯끼리는 같은 분할이라 비교 가능."
echo "=========================================================="
exit "$TRAIN_STATUS"
