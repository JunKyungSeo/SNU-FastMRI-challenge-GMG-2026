#!/usr/bin/env bash
# ============================================================
# GPU 작업 큐 슬롯 — MRAugment A/B 실험, 증강 ON arm (promptmr_kfold_f2_aug)   [shell22.sh]
#
#   [실험 목적] 데이터 증강(MRAugment)이 val loss 과적합 구간을 늦추는지 보는 A/B 실험이다.
#     - OFF arm = 이미 완료된 shell21.sh 런(net_name=promptmr_kfold_f${FOLD}). 여기서 다시
#       돌리지 않는다 — 21시간짜리 런을 또 태울 이유가 없다.
#       ⚠️ 이 런의 완주·과적합 여부는 repo에 기록 없음 — 재사용 전 config 대조 필수.
#     - ON  arm = 이 슬롯(shell22.sh). train.py 호출은 shell21.sh와 **증강 인자 7개와
#       -n/--run-tag(NET/TAG) 이름만 다르다.** FOLD·K_FOLD·KFOLD_SEED·num-cascades·n-feat0·
#       sens-n-feat0·n-history·lr·fg-loss·area-weight·grad-clip·accum-steps·데이터 경로는
#       baseline과 완전히 동일하게 유지한다 — 변경점을 1개(증강 유무)로 좁혀야 A/B가
#       과학적으로 의미가 있다.
#
#   [판정 방법] ON arm 학습 후 baseline과 비교할 것:
#     ① best-epoch val loss 값 자체(더 낮으면 증강이 도움) — val_loss_log_pmr_kf_f${FOLD}_aug.npy
#     ② val loss가 꺾이는(과적합으로 도는) epoch 인덱스가 baseline보다 뒤로 밀렸는가
#        (증강이 과적합 시점을 늦추는 게 이 실험의 핵심 가설)
#     ③ 두 로그를 나란히 그려서 곡선 모양 비교(초반 수렴 속도가 크게 나빠지지 않았는가)
#     ※ recon_eval(리더보드 채점)은 여기서 하지 않는다 — leaderboard 데이터가 있는 다른
#       환경에서 best_model.pt를 옮겨 나중에 돌린다(GPU 시간·디스크 절약, shell21.sh와 동일 방침).
#
#   [AUG_DELAY] --aug-delay 값은 "baseline val loss가 최저를 찍은 epoch − 1"이다(증강 확률
#     램프가 과적합이 시작되는 구간을 덮게 하는 것이 목적). 이 값은 사람이 baseline의
#     val_loss_log*.npy를 읽어서 결정해야 한다 — 코드가 추정하거나 기본값을 넣지 않는다.
#     AUG_DELAY 환경변수가 없으면 본 학습은 즉시 멈추고 아래에서 구하는 방법을 안내한다
#     (SMOKE=1일 때는 무관 — 스모크는 항상 constant+delay=0을 쓴다. 아래 SMOKE 분기 참고).
#
#   ⚠️ f1(promptmr_kfold_f1)은 병합 전 '이름 그룹' 분할로 학습되어 지금(volume 단위) 분할과
#      hold-out이 다르다. f1과는 val loss를 나란히 비교하지 말 것 — f2 이후 슬롯끼리만
#      (즉 이 ON arm은 f2 이후의 baseline과만) 비교 가능하다. shell21.sh 하단 주석과 동일 취지.
#
# 실행:
#   SMOKE=1 bash shell_scripting/shell22.sh              # 120 iter — 먼저 이것. p>0 확인 전엔 본 학습 금지.
#   AUG_DELAY=<값> bash shell_scripting/shell22.sh       # 본 학습 10 epoch (baseline 기준 ~21시간)
#   AUG_DELAY=<값> FOLD=3 bash shell_scripting/shell22.sh  # 다른 fold를 돌릴 때
#   FORCE=1 ...                                          # 기존 결과 폴더를 덮어쓸 때만 (기본은 거부)
# ============================================================
set -uo pipefail
export PYTHONUNBUFFERED=1

# --- repo 루트로 이동 ---
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.." || exit 1

# --- K-Fold 설정 (train.py --kfold-seed 기본값 430과 맞출 것 — baseline과 동일해야 한다) ---
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
    NET="promptmr_kfold_f${FOLD}_aug_smoke"; TAG="pmr_kf_f${FOLD}_aug_smoke"; EXTRA="--max-iters 120"; EPOCHS=1
    # ⚠️ 이거 "단순화"하지 말 것 — 일관성 없어 보이지만 의도한 것이다.
    #    1-epoch 스모크(epoch=0)에서 exp/ramp 스케줄은 공식상 t=delay일 때 1 - e^0 = 0, 즉
    #    p(epoch=0) = 0 이 **정확히** 나온다. 그러면 스모크 런은 "[aug] epoch=0 p=0.0000"을 찍고
    #    p<=0 조기 리턴으로 증강을 0회 수행한 채 exit 0으로 "성공"해버려서, 이 스모크 게이트가
    #    아무것도 검증하지 못한 채 통과한 것처럼 보인다. 그래서 스모크에서만 스케줄 자체를
    #    constant로 바꾸고 delay=0을 줘서 p = p_max = 0.55 를 epoch 0부터 강제한다.
    #    본 학습(SMOKE=0, 아래 else)은 이 override와 무관하게 항상 exp + AUG_DELAY를 쓴다.
    AUG_SCHEDULE="constant"; AUG_DELAY=0
else
    NET="promptmr_kfold_f${FOLD}_aug";       TAG="pmr_kf_f${FOLD}_aug";       EXTRA="";                EPOCHS=10
    AUG_SCHEDULE="exp"
    if [ -z "${AUG_DELAY:-}" ]; then
        echo "[STOP] AUG_DELAY가 설정되지 않았다 — baseline(promptmr_kfold_f${FOLD})의 val loss가"
        echo "       최저를 찍은 epoch을 사람이 직접 읽어서 넘겨야 한다(자동 추정/기본값 금지)."
        echo "       서버에서 아래 명령으로 값을 구한 뒤 AUG_DELAY=<값> 으로 재실행할 것:"
        cat <<EOF
python -c "
import numpy as np, sys
log = np.load(sys.argv[1])
for e, v in log: print(int(e), f'{v:.6f}')
best = int(log[np.argmin(log[:,1]), 0])
print('best epoch =', best, ' ← D는 이 값보다 작아야 한다(상한으로만 쓴다)')
print('권장 AUG_DELAY =', min(3, max(0, best - 1)), ' # 10 epoch 런은 D=2~3. best-1을 그대로 쓰면 증강이 1~2 epoch만 돌아 실험이 답을 못 준다')
" ../result/promptmr_kfold_f${FOLD}/val_loss_log_pmr_kf_f${FOLD}.npy
EOF
        exit 2
    fi
fi

echo "=========================================================="
echo "[START] $SLOT (fold $FOLD / $K_FOLD)   $(date '+%F %T')   SMOKE=$SMOKE  NET=$NET"
echo "[CWD  ] $(pwd)"
echo "[LOG  ] $LOG"
echo "[CONF ] full coil · nc10 · nf48 · sens24 · lr 2e-4 · K=$K_FOLD · fold=$FOLD · seed=$KFOLD_SEED · ${EPOCHS} epoch"
echo "[AUG  ] schedule=$AUG_SCHEDULE delay=$AUG_DELAY max-prob=0.55 exp-decay=5.0 mask-thin-max=4 orig-mask-prob=0.5"
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

# --- 가드: 기존 학습 결과를 말없이 덮어쓰지 않는다 (baseline도 21시간짜리다) ---
if [ -e "../result/$NET/checkpoints/model.pt" ] && [ "${FORCE:-0}" != "1" ]; then
    echo "[STOP] ../result/$NET/checkpoints/model.pt 가 이미 있다 — 덮어쓰면 이전 학습 결과가 사라진다."
    echo "       이어서 학습하려면 train.py에 --resume 을 붙이고, 정말 새로 덮어쓸 거면 FORCE=1 로 실행할 것."
    exit 2
fi

# ==========================================================
#  ↓↓↓  PromptMR+ K-Fold 학습 + MRAugment ON (fold $FOLD)  ↓↓↓
#     baseline(shell21.sh)과 train.py 인자는 --aug-* 7개와 -n/--run-tag(NET/TAG)만 다르다.
#     이 슬롯도 학습 + hold-out 검증까지만 한다. 채점(recon_eval)은 leaderboard 데이터가
#     있는 다른 환경에서 하므로 여기서 GPU를 더 쓰지 않는다.
# ==========================================================
set -x
python train.py --model promptmr --readout-crop \
    --num-cascades 10 --n-feat0 48 --sens-n-feat0 24 --n-history 3 \
    -b 1 -l 2e-4 -e "$EPOCHS" -r 100 --num-workers 4 --use-checkpoint \
    --fg-loss --area-weight --grad-clip 0.1 --accum-steps 2 \
    --k-fold "$K_FOLD" --val-fold "$FOLD" --kfold-seed "$KFOLD_SEED" $EXTRA \
    --aug-on --aug-schedule "$AUG_SCHEDULE" --aug-max-prob 0.55 --aug-delay "$AUG_DELAY" \
    --aug-exp-decay 5.0 --aug-mask-thin-max 4 --aug-orig-mask-prob 0.5 \
    -n "$NET" --run-tag "$TAG" \
    -t /workspace/Data/train/ -v /workspace/Data/val/
{ TRAIN_STATUS=$?; set +x; } 2>/dev/null
echo "[TRAIN] $NET (fold $FOLD) 종료   exit=$TRAIN_STATUS"

if [ "$SMOKE" = "1" ]; then
    echo "=========================================================="
    echo "[SMOKE] 확인할 것 (fold $FOLD, 증강 ON arm):"
    echo "        (0) ⚠️ 가장 중요 — 로그에 '[aug] epoch=0 p=0.5500' (또는 p>0인 값)이 찍혔는가."
    echo "            p=0.0000 이면 STOP — 본 학습(21시간)을 시작하지 말 것. 증강이 조용히"
    echo "            0회 수행되고 있다는 뜻이고, 그 상태로 21시간을 태우면 아무것도 검증하지 못한다."
    echo "        (1) OOM 없이 exit 0 인가"
    echo "        (2) iter 0 대비 Loss 가 내려가는가"
    echo "        (3) 'Loss = ... Time = ...' per-iter 시간을 baseline(promptmr_kfold_f${FOLD}) 로그와"
    echo "            비교할 것. 증강은 샘플당 FFT 2회를 dataloader에 추가한다 — --num-workers 4에서"
    echo "            CPU 병목이면 21시간이 더 늘어난다. 증가폭이 크면 --num-workers 상향 후 재측정."
    echo "        (0)이 실패하면 본 학습을 시작하지 말 것. 통과하면:"
    echo "        AUG_DELAY=<baseline에서 구한 값> bash shell_scripting/shell22.sh  로 본 학습."
    echo "=========================================================="
fi

echo "=========================================================="
echo "[DONE ] $SLOT (fold $FOLD)   $(date '+%F %T')   train=$TRAIN_STATUS"
echo "[저장] val loss: ../result/$NET/val_loss_log_${TAG}.npy"
echo "       가중치  : ../result/$NET/checkpoints/best_model.pt   ← 채점은 이걸 다른 환경으로 옮겨서"
echo "[비교] OFF arm = ../result/promptmr_kfold_f${FOLD}/val_loss_log_pmr_kf_f${FOLD}.npy (재학습 안 함)"
echo "       ⚠️ f1(promptmr_kfold_f1)은 병합 전 '이름 그룹' 분할이라 hold-out이 다르다 — f1과는"
echo "       나란히 비교하지 말 것. f2 이후 슬롯끼리(이 ON arm 포함)만 비교 가능."
echo "=========================================================="
exit "$TRAIN_STATUS"
