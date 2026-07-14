#!/usr/bin/env bash
# ============================================================
# GPU 작업 큐 슬롯  —  GPU를 쓰는 명령은 이 폴더 스크립트에서만 실행한다.
# 목적: B5(foreground 마스크 손실·검증, --fg-loss) A/B의 ON-팔 = run11.
#       run10과 "완전히 동일한 설정 + --fg-loss만 추가"로 학습한다. (OFF-팔 = run10.)
#
#   [왜 -e 10 인가 — 절대 -e 2로 바꾸지 말 것]
#     비교 대상(off-팔)은 run10의 epoch2 체크포인트(= result/sdum_run10_off2ep, 이미 복사해 둠)다.
#     cosine LR 스케줄은 total_iters = num_epochs × len 으로 잡히므로, run11도 -e 10 이어야
#     epoch0~1의 LR 곡선이 run10과 '완전히 동일'해진다. -e 2로 줄이면 cosine이 2 epoch 만에
#     0으로 붕괴해 초반 LR이 달라지고, A/B 차이가 '손실 차이'가 아니라 'LR 차이'로 오염된다.
#     → 스케줄은 10 epoch로 두되, 부분 A/B는 epoch2까지만 필요하다(아래).
#
#   [부분 A/B 절차 — epoch2까지만]
#     1) 이 슬롯 실행(run10과 동일 seed 430·데이터 순서, 손실만 --fg-loss).
#     2) 이 스크립트가 val_loss_log_run11.npy 2줄(=2에폭 완료·~11h)을 감지하면 학습을 자동 종료한다(self-stop).
#        그 시점 model.pt = epoch2 체크포인트(= off-팔 sdum_run10_off2ep 과 매칭).
#        ※ 풀 비교(10 epoch)를 원하면 아래 STOP_AFTER_EPOCHS 값을 크게(예: 99) 바꾸면 됨.
#     3) 채점(둘 다 recon_eval, 고정 채점기):
#          python recon_eval.py -n sdum_run10_off2ep  -p /root/Data/leaderboard   # off-팔(run10 epoch2)
#          python recon_eval.py -n sdum_run11         -p /root/Data/leaderboard   # on-팔(run11 epoch2 = best_model.pt)
#        비교: SSIM_full(오르나?) · SSIM_bbox(안 떨어지나=게이트) · 추론시간(불변이어야).
#     ※ val 숫자로 직접 비교 금지 — run10=전체이미지, run11=masked라 자가 다르다. 판정은 recon_eval.
#
#   [주의] run10이 GPU 점유 중이면 이 슬롯은 큐로 그 뒤에 돈다.
#
# 실행:  bash shell10.sh                         (단독)
#        ... ; bash shell10.sh                    (큐로 이어서, run10 뒤에)
# ============================================================
set -uo pipefail
export PYTHONUNBUFFERED=1   # tee 파이프로 가도 python 출력을 실시간으로 (버퍼링 방지)

# --- repo 루트로 이동 (train.py 등은 repo 루트 기준으로 돈다) ---
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.." || exit 1

# --- 로그 파일 (이름에 timestamp) ---
SLOT="$(basename "$0" .sh)"
TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="shell_scripting/logs"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/${SLOT}_${TS}.log"

# stdout + stderr 를 터미널과 로그에 동시에 남긴다
exec > >(tee -a "$LOG") 2>&1

echo "=========================================================="
echo "[START] $SLOT   $(date '+%F %T')"
echo "[CWD  ] $(pwd)"
echo "[LOG  ] $LOG"
echo "=========================================================="

# --- cv2 보장: --fg-loss는 transforms → metrics(cv2)를 import하므로 학습에도 cv2 필요.
#     libGL 없는 서버라 headless. 이미 되면 no-op. ---
python -c "import cv2" 2>/dev/null || pip install -q opencv-python-headless

# ==========================================================
#  ↓↓↓  실제 GPU 명령 (run11 = run10 + --fg-loss)  ↓↓↓
# ==========================================================
# ---- self-stop 설정: N에폭 완료되면 학습 자동 종료 (부분 A/B용) ----
#   -e 10 은 그대로 둔다(초반 LR 스케줄을 run10과 동일하게 맞추기 위함). 실제 정지는 아래 감시 루프가 한다.
STOP_AFTER_EPOCHS=10
VALLOG="/root/result/sdum_run11/val_loss_log_run11.npy"   # 완료 epoch 1개당 1줄 append됨

set -x
# run11 = run10과 완전히 동일(모델·cascade·channels·LR·epoch·warmup·readout-crop·seed 기본430) + --fg-loss.
# (README 규칙6: 순차 run 번호 필수·재사용 금지 → run10 다음이라 run11.)  백그라운드로 띄우고 감시 루프가 관리.
python train.py --model sdum --num-cascades 4 --channels 16 32 \
    -b 1 -l 2.4e-4 -e 10 -r 100 --num-workers 4 \
    --readout-crop --warmup-iters 200 --fg-loss \
    -n sdum_run11 --run-tag run11 -t /root/Data/train/ -v /root/Data/val/ &
TRAIN_PID=$!
{ set +x; } 2>/dev/null

# echo "[self-stop] train PID=$TRAIN_PID — ${STOP_AFTER_EPOCHS}에폭 완료 시 자동 종료 (감시: $VALLOG)"
# STOPPED_OK=0
# while kill -0 "$TRAIN_PID" 2>/dev/null; do
#     rows=$(python -c "import numpy as np,os; p='$VALLOG'; print(np.load(p).shape[0] if os.path.exists(p) else 0)" 2>/dev/null || echo 0)
#     if [ "${rows:-0}" -ge "$STOP_AFTER_EPOCHS" ]; then
#         echo "[self-stop] ${rows}에폭 완료 감지 → 체크포인트 저장 마무리 대기(30s) 후 종료"
#         sleep 30                                    # np.save 뒤 save_model(model.pt=epoch2)까지 완료 대기
#         kill -INT "$TRAIN_PID" 2>/dev/null           # KeyboardInterrupt → torch가 dataloader 워커·CUDA 정리
#         sleep 15
#         if kill -0 "$TRAIN_PID" 2>/dev/null; then    # 안 죽으면 강제 종료 + 남은 워커 정리
#             kill -9 "$TRAIN_PID" 2>/dev/null
#             pkill -9 -P "$TRAIN_PID" 2>/dev/null
#         fi
#         STOPPED_OK=1
#         break
#     fi
#     sleep 60
# done

# wait "$TRAIN_PID" 2>/dev/null; RC=$?
# # 의도한 2에폭 정지면 성공(0), 그 전에 스스로 죽었으면(크래시) 실제 코드 보고.
if [ "$STOPPED_OK" = 1 ]; then STATUS=0; else STATUS=$RC; fi
echo "[self-stop] 학습 종료(exit=$STATUS). run11 epoch${STOP_AFTER_EPOCHS} 체크포인트: ../result/sdum_run11/checkpoints/{best_model,model}.pt"
echo "[self-stop] 채점:  python recon_eval.py -n sdum_run10_off2ep -p /root/Data/leaderboard   # off-팔"
echo "[self-stop]        python recon_eval.py -n sdum_run11        -p /root/Data/leaderboard   # on-팔(run11)"

# ==========================================================
#  ↑↑↑  실제 GPU 명령 끝 (STATUS 는 위 self-stop 블록에서 설정)  ↑↑↑
# ==========================================================

echo "=========================================================="
echo "[DONE ] $SLOT   $(date '+%F %T')   exit=$STATUS"
echo "=========================================================="
exit "$STATUS"
