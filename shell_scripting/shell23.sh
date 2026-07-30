#!/usr/bin/env bash
# ============================================================
# GPU 작업 큐 슬롯 — recon_eval 추론 VRAM 탐침                        [shell23.sh]
#
# 목적: **약한 GPU(4GB 등)에서 채점(recon_eval)이 돌아가는지**를 숫자로 판정한다.
#   학습이 아니라 추론이라 no_grad 하에서 돌고 backward 그래프를 안 쌓는다.
#   학습 peak(3.35GB)와 비교하면 안 되고, 이 슬롯이 찍는 값이 실제 채점 요구량이다.
#
#   측정 대상 4가지 (각각 별도 프로세스 = VRAM 깨끗이 초기화):
#     ① 단일 모델 · checkpoint off  ← test_part.load_model 의 현재 기본값
#     ② 단일 모델 · checkpoint on
#     ③ MoE(전문가 2개) · checkpoint off
#     ④ MoE · checkpoint on
#
#   [왜 checkpoint on/off 를 재나]
#     test_part.load_model:36,125 이 추론에서 `model.use_checkpoint = False` 로 꺼둔다.
#     켜면 cascade 단위로 메모리를 재사용해 peak 가 내려가고 대신 느려진다. 채점 시간을
#     신경 안 쓰는 환경(원 서버가 아닌 곳에서 SSIM 두 축만 볼 때)이라면 사실상 공짜 레버라
#     양쪽을 다 재서 4GB 에 들어가는 쪽을 고른다.
#     ⚠️ 단 추론시간은 채점 항목이다(0.5·full + 0.5·bbox + 시간). checkpoint 를 켜거나 약한
#        GPU 로 잰 시간은 제출용 숫자가 아니다 — SSIM 두 축만 보는 용도로 쓸 것.
#
#   [입력] 이 데이터셋 최악 폭 슬라이스 W=480 (train/knee_acc4_46.h5, 38 slice).
#          전체의 0.7% 뿐이지만 여기서 안 터지면 나머지(W=368/372)는 여유가 있다.
#
# 실행:  bash shell_scripting/shell23.sh              (GPU 빈 상태에서, 1~2분)
#        NETS="promptmr_moe" bash shell_scripting/shell23.sh   (일부만 재고 싶을 때)
# ============================================================
set -uo pipefail
export PYTHONUNBUFFERED=1

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.." || exit 1

SLOT="$(basename "$0" .sh)"
TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="shell_scripting/logs"; mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/${SLOT}_${TS}.log"
exec > >(tee -a "$LOG") 2>&1

NETS="${NETS:-promptmr_kfold_f2 promptmr_moe}"
KSPACE="${KSPACE:-/workspace/Data/train/kspace/knee_acc4_46.h5}"

echo "=========================================================="
echo "[START] $SLOT   $(date '+%F %T')"
echo "[CWD  ] $(pwd)   [LOG] $LOG"
echo "[대상 ] $NETS   [입력] $KSPACE"
echo "=========================================================="

# --- 가드: GPU 비었는지 ---
USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
echo "[GPU  ] used=${USED}MiB"
if [ -n "$USED" ] && [ "$USED" -ge 500 ]; then
    echo "[STOP] GPU가 사용 중(${USED}MiB)이다. 빈 상태에서 실행할 것."
    exit 2
fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | sed 's/^/[GPU  ] /'

for NET in $NETS; do
    if [ ! -f "../result/$NET/checkpoints/best_model.pt" ]; then
        echo "[SKIP ] $NET — 체크포인트 없음"
        continue
    fi
    for CK in "" "--use-checkpoint"; do
        LABEL="$NET  checkpoint=$([ -z "$CK" ] && echo off || echo on)"
        echo ""
        echo "---------- $LABEL ----------"
        # 프로세스를 매번 새로 띄운다 — VRAM 을 깨끗한 상태에서 재기 위해.
        python scripts/probe_recon_vram.py -n "$NET" -p "$KSPACE" $CK
        echo "[exit ] $?"
    done
done

echo ""
echo "=========================================================="
echo "[DONE ] $SLOT   $(date '+%F %T')"
echo "[읽는 법] 각 조합의 'peak reserved' 가 목표 GPU VRAM 보다 작으면 그 조합으로 채점 가능."
echo "          4GB 에서 안 되면: checkpoint on → 그래도 안 되면 단일 모델 → 그래도 안 되면"
echo "          CUDA_VISIBLE_DEVICES=\"\" 로 CPU 채점(느리지만 VRAM 무관, recon_eval 수정 불필요)."
echo "=========================================================="
