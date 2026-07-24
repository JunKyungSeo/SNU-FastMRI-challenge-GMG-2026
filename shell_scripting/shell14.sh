#!/usr/bin/env bash
# ============================================================
# GPU 작업 큐 슬롯  —  GPU를 쓰는 명령은 이 폴더 스크립트에서만 실행한다.
# 목적: PromptMR+ 후속 — 두 단계를 순서대로.
#   [1] hspromptmr_run1 공식 채점(recon_eval, 3축). shell13 이 -n promptmr_run1(오타)
#       로 [skip] 났었음 → 실제 체크포인트 이름 hspromptmr_run1 로 첫 실점수 확보.
#   [2] run2 학습 = run1 과 동일 tiny config 에 개선 2개만 추가:
#        · --fg-loss        : 손실·검증을 채점 정합 foreground 마스크 안에서만
#                             (VarNet run10→run11 에서 이 한 개로 full+0.0043·bbox+0.0074, 3축↑)
#        · --warmup-iters 200 → cosine : run11 우승 LR 레시피(상수→cosine tail).
#       run1 은 아직 ep8 에서 NewRecord(과소학습) → 10ep 재학습이 바닥.
#   ⚠️ [1] 채점이 실패/skip 해도 [2] 학습은 계속 진행된다(아래 STATUS=학습 종료코드).
#   ⚠️ metrics.py 가 cv2 import → libGL 없는 서버는 opencv-python-headless 필요(아래 보장).
#   ⚠️ 이름은 맨 아래 변수(RUN1_NAME/RUN2_NAME)로 통제 — 실제 run1 이름이 다르면 여기만 고친다.
#
# 실행:  bash shell14.sh                          (단독)
#        bash shell13.sh ; bash shell14.sh        (큐로 이어서)
# ============================================================
set -uo pipefail
export PYTHONUNBUFFERED=1   # tee 파이프로 가도 python 출력을 실시간으로 (버퍼링 방지)

# --- repo 루트로 이동 (train.py / recon_eval.py 는 repo 루트 기준으로 돈다) ---
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

# --- 시작 고지 (어느 슬롯이 어디서 시작됐는지) ---
echo "=========================================================="
echo "[START] $SLOT   $(date '+%F %T')"
echo "[CWD  ] $(pwd)"
echo "[LOG  ] $LOG"
echo "=========================================================="

# --- run 이름 (실제 서버 산출물 이름에 맞춘다) ---
RUN1_NAME="hspromptmr_run1"     # [1] 채점 대상 = 이미 학습된 기존 run
RUN2_NAME="hspromptmr_run2"     # [2] 새로 돌릴 run

# --- cv2 보장 (metrics.py import용, libGL 없는 서버 → headless). 이미 되면 no-op ---
python -c "import cv2" 2>/dev/null || pip install -q opencv-python-headless

# ==========================================================
#  [1] ↓↓↓  기존 run1 공식 채점 (recon_eval, 3축)  ↓↓↓
#      load_model 이 best_model.pt 의 저장 args(model==promptmr)로 config 자동 복원
#      → --model/--num-cascades 등 불필요. 실패해도 학습([2])은 계속.
# ==========================================================
echo "[STEP 1] recon_eval  -n $RUN1_NAME"
set -x
if [ -f "../result/${RUN1_NAME}/checkpoints/best_model.pt" ]; then
    python recon_eval.py -n "$RUN1_NAME" -p /root/Data/leaderboard || \
        echo "[warn] recon_eval($RUN1_NAME) 실패 — 학습은 계속 진행."
else
    echo "[skip] ../result/${RUN1_NAME}/checkpoints/best_model.pt 없음 — 이름 확인(RUN1_NAME)."
fi
{ set +x; } 2>/dev/null

# ==========================================================
#  [2] ↓↓↓  run2 학습 (tiny config + fg-loss + warmup→cosine, 10ep)  ↓↓↓
#      run1 대비 변경점은 딱 2개: --fg-loss, --warmup-iters 200.
#      나머지(num-cascades 6·n-feat0 16·channels 24 32 40·readout-crop·lr 2e-4·batch1)는 동일.
# ==========================================================
echo "[STEP 2] train  -n $RUN2_NAME  (--fg-loss + warmup200→cosine, 10ep)"
set -x
python train.py --model promptmr --num-cascades 6 --n-feat0 16 --channels 24 32 40 \
    --readout-crop -b 1 -l 2e-4 -e 10 -r 100 --num-workers 4 \
    --fg-loss --warmup-iters 200 \
    --run-tag "$RUN2_NAME" -n "$RUN2_NAME" -t /root/Data/train/ -v /root/Data/val/

# ==========================================================
#  ↑↑↑  실제 GPU 명령 끝 (STATUS 는 마지막 명령=학습의 종료코드)  ↑↑↑
# ==========================================================
{ STATUS=$?; set +x; } 2>/dev/null

echo "=========================================================="
echo "[DONE ] $SLOT   $(date '+%F %T')   exit=$STATUS"
echo "[확인] STEP1: run1 3축(SSIM_full/bbox/time) 로그 · STEP2: fg-val 감소 + best_model.pt 저장."
echo "[다음] STEP2 끝나면 shell13 방식으로 $RUN2_NAME 채점 → training_anal 에 run11(VarNet)·run1 대비 기록."
echo "=========================================================="
exit "$STATUS"
