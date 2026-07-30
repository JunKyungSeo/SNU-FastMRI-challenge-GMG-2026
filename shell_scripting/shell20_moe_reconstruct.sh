#!/usr/bin/env bash
# ============================================================
# GPU 작업 큐 슬롯 — promptmr_moe (acc4/acc8 MoE) reconstruct + 채점 (SSIM_full/bbox/시간).
#   promptmr_moe/checkpoints/best_model.pt = merge_moe.py 로 만든 MoE 병합 체크포인트
#   구조: {moe:True, experts:{acc4,acc8}, router:{threshold}}.
#   load_model 이 마스크의 가속률로 acc4/acc8 전문가를 골라 라우팅(MoEWrapper) → --model 등 인자 불필요.
#   ⚠️ MoE 코드 있는 브랜치(jkMRplus, utils/model/moe.py 존재)에서 실행할 것.
#
# ⚠️ 출력: recon_eval.py 는 acc4·acc8 을 '전부 처리한 맨 끝'에서 점수를 한 번에 print
#   (진행 중엔 조용함 = 정상). PYTHONUNBUFFERED=1 + `python -u` 로 tee 파이프로도 실시간 표시.
#
# 실행:  bash shell_scripting/shell20_moe_reconstruct.sh   (GPU 빈 상태에서)
# ============================================================
set -uo pipefail
export PYTHONUNBUFFERED=1   # tee 파이프로 가도 python 출력을 실시간으로 (버퍼링 방지)

# --- repo 루트로 이동 (recon_eval.py 는 repo 루트, '../result' 기준) ---
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.." || exit 1

# --- 로그 파일 (이름에 timestamp) ---
SLOT="$(basename "$0" .sh)"
TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="shell_scripting/logs"; mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/${SLOT}_${TS}.log"
exec > >(tee -a "$LOG") 2>&1

NET="promptmr_moe"
LEADERBOARD="${LEADERBOARD:-/root/Data/leaderboard}"
CKPT="../result/${NET}/checkpoints/best_model.pt"
OUT="../result/${NET}/recon_eval_moe_${TS}.txt"   # 점수 사본(timestamp 로 덮어쓰기 방지)

echo "=========================================================="
echo "[START] $SLOT   $(date '+%F %T')   NET=$NET"
echo "[BR   ] $(git branch --show-current 2>/dev/null) ($(git rev-parse --short HEAD 2>/dev/null))"
echo "[LOG  ] $LOG"
echo "=========================================================="

# --- 가드: MoE 코드 (없으면 라우팅 불가 → KeyError) ---
[ -f utils/model/moe.py ] || { echo "[STOP] utils/model/moe.py 없음 — MoE 코드 있는 브랜치로: git checkout jkMRplus"; exit 2; }

# --- 가드: 체크포인트 존재 ---
[ -f "$CKPT" ] || { echo "[STOP] $CKPT 없음 — merged best_model.pt 를 여기 업로드할 것."; exit 2; }

# --- 가드: 진짜 MoE 병합 파일인지 (업로드 깨짐/미완/오파일 fail-fast) ---
python - "$CKPT" <<'PYCHK'
import sys, torch
p = sys.argv[1]
try:
    ck = torch.load(p, map_location='cpu', mmap=True, weights_only=False)
except Exception as e:
    print(f"[STOP] 체크포인트를 못 읽는다({type(e).__name__}) — 업로드가 깨졌거나 미완이다. 다시 올릴 것.")
    sys.exit(2)
if not (isinstance(ck, dict) and ck.get('moe') and 'experts' in ck):
    keys = list(ck.keys()) if isinstance(ck, dict) else type(ck).__name__
    print(f"[STOP] MoE 병합 파일이 아니다(moe/experts 키 없음). 최상위={keys}. 올바른 merged 파일 확인.")
    sys.exit(2)
th = ck.get('router', {}).get('threshold')
note = "(fix됨)" if th == 0.25 else "  ⚠️(0.25 아님 — acc8 라우팅 주의, 0.1875면 재-merge 권장)"
print(f"[OK] MoE 검증 통과: experts={list(ck['experts'].keys())}  router.threshold={th} {note}")
PYCHK
GUARD=$?; [ "$GUARD" -ne 0 ] && exit "$GUARD"

# --- 가드: GPU 빈지 (MoE 는 전문가 2개라 무겁다 → OOM 방지). 강행하려면 이 블록 주석 처리 ---
USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
echo "[GPU ] used=${USED}MiB"
if [ -n "$USED" ] && [ "$USED" -ge 500 ]; then echo "[STOP] GPU 사용 중(${USED}MiB) — 빈 상태에서 실행(MoE 는 전문가 2개라 OOM 위험)."; exit 2; fi

# --- 가드: cv2 (metrics.foreground_mask 가 import) ---
python -c "import cv2" 2>/dev/null || { echo "[STOP] cv2 import 실패 — bash /root/restore_env.sh 후 재실행."; exit 2; }

echo "[NOTE] recon_eval 은 끝에 점수를 한 번에 출력(진행 중엔 조용함=정상). 점수 사본: $OUT"

# ==========================================================
#  ↓↓↓  promptmr_moe (진짜 MoE) reconstruct + 채점  ↓↓↓
# ==========================================================
set -x
python -u recon_eval.py -n "$NET" -p "$LEADERBOARD" | tee "$OUT"
{ STATUS=${PIPESTATUS[0]}; set +x; } 2>/dev/null   # tee 가 아니라 python 종료코드

echo "=========================================================="
echo "[DONE ] $SLOT   $(date '+%F %T')   recon_eval exit=$STATUS"
echo "[저장] 점수: $OUT   (+ 로그: $LOG)"
echo "[비교] shared baseline: full 0.9309 / bbox 0.9284 (acc8 0.9131/0.9074) — MoE 가 이걸 넘어야 의미."
echo "=========================================================="
exit "$STATUS"
