#!/usr/bin/env bash
# ============================================================
# GPU 작업 큐 슬롯  —  GPU를 쓰는 명령은 이 폴더 스크립트에서만 실행한다.
# 목적: PromptMR+ MVP build/shape/OOM probe. num_cascades=2 + --max-iters 2 로
#       forward/backward 가 8GB에 들어가는지, 출력이 (1,384,384) 인지, 첫 val SSIM 이
#       ~0 이 아닌지(스케일 정합)만 본다. 이건 일회용 검증(로그되는 run 아님).
#       ⚠️ OOM 나면 레버 순서(spec §6): --readout-crop(이미 ON) → coil-compress →
#          --num-cascades 감소 → --n-feat0 8. peak CUDA mem 을 로그에서 확인.
#
# 실행:  bash shell11.sh                         (단독)
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
#  ↓↓↓  실제 GPU 명령 (PromptMR+ probe)  ↓↓↓
#  set -x 로 실행되는 명령을 로그에 그대로 남긴다.
# ==========================================================
set -x
python train.py --model promptmr --num-cascades 2 --n-feat0 16 --channels 24 32 40 \
    --readout-crop -b 1 -l 2e-4 -e 1 --max-iters 2 -r 1 --num-workers 4 \
    -n promptmr_probe -t /root/Data/train/ -v /root/Data/val/

# ==========================================================
#  ↑↑↑  실제 GPU 명령 끝 (STATUS 는 마지막 명령의 종료코드)  ↑↑↑
# ==========================================================
{ STATUS=$?; set +x; } 2>/dev/null

echo "=========================================================="
echo "[DONE ] $SLOT   $(date '+%F %T')   exit=$STATUS"
echo "[확인] 출력 shape (1,384,384) · OOM 없음 · 첫 val SSIM≠0 · peak mem."
echo "[다음] OK 면 shell12(run1 smoke). OOM 이면 spec §6 레버 순서로 조정."
echo "=========================================================="
exit "$STATUS"
