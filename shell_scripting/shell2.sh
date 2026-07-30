#!/usr/bin/env bash
# ============================================================
# GPU 작업 큐 슬롯  —  GPU를 쓰는 명령은 이 폴더 스크립트에서만 실행한다.
# 목적: run12(shell3, bbox+UC) 본run 전 게이트 smoke. UC+bbox(+fg) 조합이 실데이터에서
#       크래시 없이 3-iter 돌고 체크포인트에 uc=True 가 저장되는지만 확인(품질 X, crash-check).
#       ⚠️ UC는 GPU 전용(train.py가 .cuda() 하드코딩) → CPU smoke 불가.
#       ⚠️ disposable smoke라 결과(sdum_uc_bbox_smoke)는 확인 후 삭제. run 번호 안 씀
#           (실제 run 번호 run13+ 는 다음 실험용으로 보존).
#       통과 기준: 크래시 없이 종료 + 로그에 'SMOKE uc= True'.
#         → 통과: shell3(run12) 진행.   실패: shell4(run13 fallback).
#
# 실행:  bash shell2.sh                          (단독, shell1 recon_eval 뒤. GPU 비었을 때)
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

# --- cv2 보장: bbox/fg 경로가 metrics(cv2) 를 import. headless 서버. ---
python -c "import cv2" 2>/dev/null || { echo "[STOP] cv2 import 실패 — bash /root/restore_env.sh 로 환경 복구 후 실행할 것."; exit 2; }

# ==========================================================
#  ↓↓↓  실제 GPU 명령 (run12 와 동일 config + --max-iters 3 로 축소)  ↓↓↓
#  --fg-loss 포함 = run12 의 maximal 경로(BASELINE=run11 케이스)까지 검증. run10 케이스는 부분집합.
# ==========================================================
set -x
python train.py --model sdum --uc \
    --num-cascades 4 --channels 16 32 \
    -b 1 -l 2.4e-4 -e 1 --max-iters 3 -r 1 --num-workers 4 \
    --readout-crop --warmup-iters 200 --fg-loss --bbox-loss-lambda 0.5 \
    -n sdum_uc_bbox_smoke -t /root/Data/train/ -v /root/Data/val/

# 체크포인트에 uc/bbox 가 저장됐는지 확인 (기대: uc= True)
python -c "import torch; a=torch.load('../result/sdum_uc_bbox_smoke/checkpoints/best_model.pt', weights_only=False)['args']; print('SMOKE uc=', getattr(a,'uc',None), 'bbox_lambda=', getattr(a,'bbox_loss_lambda',None))"

# ==========================================================
#  ↑↑↑  실제 GPU 명령 끝 (STATUS 는 마지막 명령의 종료코드)  ↑↑↑
# ==========================================================
{ STATUS=$?; set +x; } 2>/dev/null

echo "=========================================================="
echo "[DONE ] $SLOT   $(date '+%F %T')   exit=$STATUS"
echo "[정리] 통과하면 결과 삭제:  rm -rf ../result/sdum_uc_bbox_smoke"
echo "[다음] 통과 → shell3 BASELINE 확인 후 bash shell_scripting/shell3.sh (run12).  실패 → shell4 (run13)."
echo "=========================================================="
exit "$STATUS"
