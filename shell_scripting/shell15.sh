#!/usr/bin/env bash
# ============================================================
# GPU 작업 큐 슬롯  —  GPU를 쓰는 명령은 이 폴더 스크립트에서만 실행한다.
# 목적: PromptMR+ 5-fold 교차검증. train/ + val/ 을 병합한 풀을 볼륨(파일) 단위로
#       5등분하고(seed 430 고정), fold 0..4 를 held-out val 로 순서대로 학습.
#       각 fold = run2 와 동일 레시피(tiny config + --fg-loss + warmup200→cosine, 10ep).
#       마지막에 fold별 best val(=1-SSIM_full)을 모아 CV mean±std 출력.
#   ⚠️ 비용: PromptMR ~1.9h/epoch × 10ep × 5fold ≈ 95시간(약 4일, 1080 한 장 순차).
#       repo 정신(augmentation_and_preprocessing.md §3.2)은 전면 k-fold를 "감당 불가"로
#       보므로, 이건 최종 후보 1개를 확정 검증할 때만 돌리는 무거운 슬롯이다.
#   ✅ 재개: fold가 끝나면 ../result/<name>/.fold_done 마커를 남긴다. 4일 사이 중단돼도
#       이 슬롯을 다시 실행하면 완료된 fold는 [skip]하고 다음 fold부터 이어 돈다.
#       특정 fold를 강제 재학습하려면 그 fold의 ../result/<name>/ 폴더를 지운다.
#   ⚠️ 볼륨 단위 분할(슬라이스 아님) → leakage 없음. 모든 fold 같은 seed → 같은 5-way 분할.
#
# 실행:  bash shell15.sh                          (단독; 아주 긴 작업)
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

# --- K-fold 파라미터 ---
K=5
SEED=430
PREFIX="hspromptmr_kf"     # fold 결과 = ../result/${PREFIX}_fold{i}/

# ==========================================================
#  ↓↓↓  실제 GPU 명령 (fold 0..K-1 순차 학습)  ↓↓↓
# ==========================================================
for i in $(seq 0 $((K - 1))); do
    NAME="${PREFIX}_fold${i}"
    if [ -f "../result/${NAME}/.fold_done" ]; then
        echo "[skip] fold ${i}/$((K - 1)) 이미 완료(.fold_done) — 건너뜀."
        continue
    fi
    echo "=========================================================="
    echo "[fold ${i}/$((K - 1))] train ${NAME}   $(date '+%F %T')"
    echo "=========================================================="
    set -x
    python train.py --model promptmr --num-cascades 6 --n-feat0 16 --channels 24 32 40 \
        --readout-crop -b 1 -l 2e-4 -e 10 -r 100 --num-workers 4 \
        --fg-loss --warmup-iters 200 \
        --kfold "$K" --fold "$i" --seed "$SEED" \
        --run-tag "$NAME" -n "$NAME" -t /root/Data/train/ -v /root/Data/val/ \
        && touch "../result/${NAME}/.fold_done"
    { set +x; } 2>/dev/null
done

# ==========================================================
#  ↓↓↓  CV 집계 (CPU, 가벼움) — fold별 best val 모아 mean±std  ↓↓↓
#  val = fg-masked SSIM 이라 1-SSIM_full. best = epoch 중 최소.
# ==========================================================
echo "=========================================================="
echo "[CV 집계]  fold별 best val(1-SSIM_full) → mean±std"
python - "$K" "$PREFIX" <<'PY'
import os, sys
import numpy as np
K = int(sys.argv[1]); prefix = sys.argv[2]
vals = []
for i in range(K):
    name = f"{prefix}_fold{i}"
    p = f"../result/{name}/val_loss_log_{name}.npy"
    if not os.path.exists(p):
        print(f"  fold{i}: (로그 없음: {p})"); continue
    a = np.load(p)
    if a.size == 0:
        print(f"  fold{i}: (로그 비어있음)"); continue
    best = float(a[:, 1].min()); ep = int(a[a[:, 1].argmin(), 0])
    vals.append(best)
    print(f"  fold{i}: best 1-SSIM={best:.4f}  SSIM_full={1-best:.4f}  (ep{ep})")
if vals:
    m, s = float(np.mean(vals)), float(np.std(vals))
    print(f"  ---")
    print(f"  CV(n={len(vals)}): 1-SSIM_full = {m:.4f} ± {s:.4f}"
          f"   |   SSIM_full = {1-m:.4f} ± {s:.4f}")
else:
    print("  집계할 fold 로그가 없음 — 학습이 하나도 완료되지 않았습니다.")
PY
STATUS=$?

echo "=========================================================="
echo "[DONE ] $SLOT   $(date '+%F %T')   exit=$STATUS"
echo "[다음] CV SSIM_full ± std 를 training_anal 에 기록(단일 split run1/run2 대비 편차 판정)."
echo "=========================================================="
exit "$STATUS"
