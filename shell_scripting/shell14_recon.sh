#!/usr/bin/env bash
# ============================================================
# GPU 작업 큐 슬롯 — hspromptmr_run2 recon_eval 채점 (3축: SSIM_full / SSIM_bbox / 시간).
#   짝 슬롯: shell14.sh 가 hspromptmr_run2 를 학습(STEP2) → 이 슬롯이 그 산출물을 채점한다.
#   load_model 이 best_model.pt 의 저장 args(model==promptmr)로 config 자동 복원
#   → --model/--num-cascades 등 인자 불필요.
#
# ⚠️ 출력: recon_eval.py 는 acc4·acc8 볼륨을 '전부 처리한 맨 끝'에서 점수를 한 번에 print 한다
#   (진행 중엔 출력이 없음 = 실행 몇 분간 화면이 조용한 건 정상, 멈춘 게 아니다).
#   PYTHONUNBUFFERED=1 + `python -u` 로 tee 파이프로도 그 최종 출력이 바로 뜨게 한다.
#   GPU 가 빈 상태에서 실행할 것(다른 학습과 동시 로드 시 8GB OOM 위험).
#
# 실행:  bash shell_scripting/shell14_recon.sh
# ============================================================
set -uo pipefail
export PYTHONUNBUFFERED=1   # tee 파이프로 가도 python 출력을 실시간으로 (버퍼링 방지)

# --- repo 루트로 이동 (recon_eval.py 는 repo 루트, '../result' 기준으로 돈다) ---
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.." || exit 1

# --- 로그 파일 (이름에 timestamp) ---
SLOT="$(basename "$0" .sh)"
TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="shell_scripting/logs"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/${SLOT}_${TS}.log"
exec > >(tee -a "$LOG") 2>&1

RUN_NAME="hspromptmr_run2"                              # 채점 대상 (shell14.sh 의 RUN2_NAME 과 동일)
OUT="../result/${RUN_NAME}/recon_eval_${TS}.txt"        # 점수 사본(timestamp 로 덮어쓰기 방지)

echo "=========================================================="
echo "[START] $SLOT   $(date '+%F %T')   NET=$RUN_NAME"
echo "[CWD  ] $(pwd)"
echo "[LOG  ] $LOG"
echo "=========================================================="

# --- cv2 보장 (metrics.py import용, libGL 없는 서버 → headless). 이미 되면 no-op ---
python -c "import cv2" 2>/dev/null || pip install -q opencv-python-headless

# --- 채점 대상 체크포인트 확인 ---
if [ ! -f "../result/${RUN_NAME}/checkpoints/best_model.pt" ]; then
    echo "[STOP] ../result/${RUN_NAME}/checkpoints/best_model.pt 없음 — 이름/학습 완료 확인(RUN_NAME)."
    exit 2
fi

echo "[NOTE] recon_eval 은 끝에 점수를 한 번에 출력한다(진행 중엔 조용함 = 정상)."
echo "[NOTE] 점수 사본: $OUT"

# ==========================================================
#  ↓↓↓  recon_eval 채점 (SSIM_full + SSIM_bbox + 시간)  ↓↓↓
# ==========================================================
set -x
python -u recon_eval.py -n "$RUN_NAME" -p /root/Data/leaderboard | tee "$OUT"
{ STATUS=${PIPESTATUS[0]}; set +x; } 2>/dev/null   # tee 가 아니라 python 의 종료코드를 잡는다

echo "=========================================================="
echo "[DONE ] $SLOT   $(date '+%F %T')   recon_eval exit=$STATUS"
echo "[저장] 점수: $OUT   (+ 로그: $LOG)"
echo "=========================================================="
exit "$STATUS"
