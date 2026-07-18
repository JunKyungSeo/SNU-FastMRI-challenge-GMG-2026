#!/usr/bin/env bash
# ============================================================
# GPU 작업 큐 슬롯  —  GPU를 쓰는 명령은 이 폴더 스크립트에서만 실행한다.
# 목적: run12 = C1(병변 bbox 가중 손실) A/B의 ON-팔.  λ=0.5, cap=2(합), 스케일 X.
#       Stage-0 진단(training_anal/2026-07-14-stage0-lesion-diagnosis.md) → 분기 C.
#       근거: ACL sprain(High/Low)·Subchondral edema·Bone-Fracture 4개 label이 "자기 슬라이스"
#       대비 -0.03~-0.05 로 유의하게 무너짐(볼륨 클러스터 부트스트랩). 오버샘플링(w)은 p=0.392라 기각.
#       ⚠️ +UC(--uc): 사용자 결정으로 UC(acc4/acc8+cascade-index 조건화)를 이 run에 함께 켰다.
#          → run12 = bbox + UC "묶음"이라 run10/run11 대비 단일-레버 A/B가 아니다(효과 분리 불가).
#          UC는 param↑→추론시간↑ → 시간축 점수 불리 가능. 게이트는 run10 실채점 3축 총점 대비로만.
#          ⚠️ UC+bbox 조합은 shell2 smoke(GPU 3-iter) 통과 후에만 본run.  UC는 GPU 전용 = CPU smoke 불가.
#
# 실행:  bash shell3.sh                          (단독, shell2 smoke 통과 후)
#
# ⚠️⚠️ 실행 전 필수 — 사람이 결정할 것 ⚠️⚠️
#   0) UC+bbox 조합 smoke 통과 확인: `bash shell_scripting/shell2.sh` (GPU 3-iter).
#      크래시 없고 로그에 'SMOKE uc= True' 찍히면 OK. (통과 전 이 run12 돌리지 말 것.)
#   1) shell1(recon_eval run11) 결과로 A/B baseline 을 정한다:
#       run11 3축이 run10(full 0.9118 / bbox 0.9021) 보다 높으면 BASELINE=run11, 아니면 run10.
#   그 결과로 아래 BASELINE 을 **명시적으로** 골라라. repo 컨벤션 = 직전 best 위에 하나만 얹는 A/B.
#       BASELINE=run11 → run11 승 (fg-loss 유지한 채 C1 추가)
#       BASELINE=run10 → run11 패 (fg-loss 빼고 C1만 추가)
#   비워두면 이 스크립트는 실행을 거부한다(임의 baseline 으로 54h 태우기 방지).
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

# --- 시작 고지 (어느 슬롯이 어디서 시작됐는지) ---
echo "=========================================================="
echo "[START] $SLOT   $(date '+%F %T')"
echo "[CWD  ] $(pwd)"
echo "[LOG  ] $LOG"
echo "=========================================================="

# ==========================================================
#  ↓↓↓  실제 GPU 명령을 여기에 작성 (이 아래만 고치면 됨)  ↓↓↓
#  set -x 로 실행되는 명령을 로그에 그대로 남긴다 (로그가 명령을 안 찍던 문제 해결)
# ==========================================================
# ---- 사람이 채울 것: shell1(recon_eval run11) 결과에 따라 baseline 선택 ----
BASELINE="run11"          # <-- "run10" 또는 "run11". 비어 있으면 실행 거부.

if [ -z "$BASELINE" ]; then
    echo "[STOP] BASELINE 이 비어 있다. shell1(recon_eval run11)을 먼저 보고 run10/run11 중 하나를 고른 뒤"
    echo "       이 스크립트의 BASELINE= 에 적어라. (임의 baseline 으로 54h를 태우지 않기 위한 가드)"
    exit 2
fi
case "$BASELINE" in
    run11) FG_FLAG="--fg-loss" ;;      # fg-loss 가 이겼다 → 유지한 채 C1만 추가
    run10) FG_FLAG="" ;;               # fg-loss 가 졌다 → 빼고 C1만 추가
    *) echo "[STOP] BASELINE 은 run10 또는 run11 이어야 한다 (받은 값: '$BASELINE')"; exit 2 ;;
esac
echo "[config] BASELINE=$BASELINE → FG_FLAG='$FG_FLAG' + C1(--bbox-loss-lambda 0.5) + UC(--uc)"

# --- cv2 보장: fg-loss/bbox 경로가 transforms → metrics(cv2) 를 import. headless 서버. ---
python -c "import cv2" 2>/dev/null || pip install -q opencv-python-headless

# λ=0.5 (사용자 확정). 손실 형태 = cap=2 합·스케일 X (loss_function.py::bbox_ssim_loss).
#   병변 슬라이스 gradient(비병변=1.00): b=1 1.55 / b=2 2.11 / b≥3 2.11(cap). 평균병변 1.73.
#   "신호부터 확인, 되면 다음 run에서 0.72 승급." A/B 게이트 = SSIM_full 비퇴행(방향이 평소와 반대).
# 그 외 조건은 run10/run11과 동일: [16,32]×4, LR 2.4e-4, 10ep, readout-crop, warmup 200, seed 430.
#   단 이번엔 --uc 가 추가돼 bbox 단독 A/B가 아니다(위 ⚠️ +UC 참고). 게이트는 run10 실채점 3축 대비.
set -x
python train.py --model sdum --num-cascades 4 --channels 16 32 \
    -b 1 -l 2.4e-4 -e 10 -r 100 --num-workers 4 \
    --readout-crop --warmup-iters 200 $FG_FLAG \
    --bbox-loss-lambda 0.5 --uc \
    -n sdum_run12 --run-tag run12 -t /root/Data/train/ -v /root/Data/val/

# ==========================================================
#  ↑↑↑  실제 GPU 명령 끝 (STATUS 는 마지막 명령의 종료코드)  ↑↑↑
# ==========================================================
{ STATUS=$?; set +x; } 2>/dev/null

echo "=========================================================="
echo "[DONE ] $SLOT   $(date '+%F %T')   exit=$STATUS"
echo "[다음] best-model(best_model.pt)로 recon_eval 채점:"
echo "    python recon_eval.py -n sdum_run12 -p /root/Data/leaderboard"
echo "  → 0.5·SSIM_full + 0.5·SSIM_bbox 로 run10(0.9118/0.9021) 대비. (val 161박스로 고르지 말 것 — leaderboard 1,229박스가 실채점)"
echo "  게이트: SSIM_bbox ↑ **그리고 SSIM_full 비퇴행**(≥ baseline). full 깎이면 λ 과함(λ↓ 또는 롤백)."
echo "  label별 재분해: scripts/bbox_ssim_breakdown.py 의 RECON 경로를 sdum_run12 로 바꿔 실행"
echo "=========================================================="
exit "$STATUS"
