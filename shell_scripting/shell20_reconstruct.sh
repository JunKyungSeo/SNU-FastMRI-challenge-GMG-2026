#!/usr/bin/env bash
# ============================================================
# GPU 작업 큐 슬롯 — promptmr_moe (acc4/acc8 MoE) reconstruct + 채점 (SSIM_full/bbox/시간).
#   짝 슬롯: shell20.sh 가 MoE 결합 체크포인트(promptmr_moe)를 만든다 → 이 슬롯이 그걸 재구성·채점한다.
#   (shell20 의 ⑤ score 단계를 독립 슬롯으로 뽑은 것. STAGE=merge 재실행 없이 채점만 다시 돌릴 때 쓴다.)
#
#   load_model 이 best_model.pt 의 'moe' 플래그를 보고 acc4/acc8 전문가를 MoEWrapper 로 세워
#   슬라이스 마스크의 가속률로 라우팅한다 → --model/--num-cascades 등 인자 불필요.
#   ⚠️ 반드시 MoE 코드가 있는 브랜치(jkMRplus @ 1db30ef 이후, utils/model/moe.py 존재)에서 실행할 것.
#      MoE 코드 없는 브랜치(예: steplr-backup)면 recon_eval 이 KeyError: 'model' 로 깨진다.
#
# ⚠️ 출력: recon_eval.py 는 acc4·acc8 볼륨을 '전부 처리한 맨 끝'에서 점수를 한 번에 print 한다
#   (진행 중엔 출력 없음 = 실행 몇 분간 화면이 조용한 건 정상, 멈춘 게 아니다).
#   PYTHONUNBUFFERED=1 + `python -u` 로 tee 파이프로도 그 최종 출력이 바로 뜨게 한다.
#
# 실행:  bash shell_scripting/shell20_reconstruct.sh        (GPU 빈 상태에서)
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

NET="promptmr_moe"
LEADERBOARD="${LEADERBOARD:-/root/Data/leaderboard}"
OUT="../result/${NET}/recon_eval_moe_${TS}.txt"   # 점수 사본(timestamp 로 덮어쓰기 방지)

echo "=========================================================="
echo "[START] $SLOT   $(date '+%F %T')   NET=$NET"
echo "[CWD  ] $(pwd)"
echo "[LOG  ] $LOG"
echo "[BR   ] $(git branch --show-current 2>/dev/null) ($(git rev-parse --short HEAD 2>/dev/null))"
echo "=========================================================="

# --- 가드: MoE 코드가 이 체크아웃에 있나 (없으면 recon_eval 이 KeyError 로 깨진다) ---
if [ ! -f utils/model/moe.py ]; then
    echo "[STOP] utils/model/moe.py 없음 — MoE 코드 없는 브랜치다(예: steplr-backup)."
    echo "       MoE 있는 브랜치로 전환 후 재실행:  git checkout jkMRplus"
    exit 2
fi

# --- 가드: 채점 대상 MoE 체크포인트 존재 ---
if [ ! -f "../result/${NET}/checkpoints/best_model.pt" ]; then
    echo "[STOP] ../result/${NET}/checkpoints/best_model.pt 없음 — shell20 의 merge(promptmr_moe 생성) 먼저."
    exit 2
fi

# --- 가드: GPU 비었는지 (used < 500MiB). MoE 는 전문가 2개라 무겁다 → 동시 로드 시 8GB OOM ---
#   강행하려면 아래 if 블록을 주석 처리(권장하지 않음).
USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
echo "[GPU ] used=${USED}MiB"
if [ -n "$USED" ] && [ "$USED" -ge 500 ]; then
    echo "[STOP] GPU 사용 중(${USED}MiB) — 빈 상태에서 실행할 것 (MoE 는 전문가 2개라 OOM 위험)."
    exit 2
fi

# --- 가드: cv2 (metrics.foreground_mask 가 cv2 import) ---
python -c "import cv2" 2>/dev/null || { echo "[STOP] cv2 import 실패 — bash /root/restore_env.sh 로 환경 복구 후 재실행."; exit 2; }

echo "[NOTE] recon_eval 은 acc4·acc8 을 다 처리한 뒤 점수를 한 번에 출력한다(진행 중엔 조용함 = 정상)."
echo "[NOTE] 점수 사본: $OUT   (+ 재구성 h5: ../result/${NET}/reconstructions_leaderboard/acc{4,8})"

# ==========================================================
#  ↓↓↓  promptmr_moe reconstruct + 채점  ↓↓↓
# ==========================================================
set -x
python -u recon_eval.py -n "$NET" -p "$LEADERBOARD" | tee "$OUT"
{ STATUS=${PIPESTATUS[0]}; set +x; } 2>/dev/null   # tee 가 아니라 python 의 종료코드를 잡는다

echo "=========================================================="
echo "[DONE ] $SLOT   $(date '+%F %T')   recon_eval exit=$STATUS"
echo "[저장] 점수: $OUT   (+ 로그: $LOG)"
echo "=========================================================="
exit "$STATUS"
