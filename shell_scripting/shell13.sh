#!/usr/bin/env bash
# ============================================================
# GPU 작업 큐 슬롯 — 코일 전부 사용(압축 없음) 메모리 탐침. 탐침만 하고 학습은 안 한다.
# 목적: coil compression 을 빼고 물리코일 15개를 그대로 쓰면서, 대신 구조적 메모리 절감으로
#       8GB 에 들어가는지 확인한다. 들어가면 cascade 를 몇 개까지 올릴 수 있는지도 같이 잰다.
#
#   왜 이 탐침이 필요한가:
#     기존 결론("코일 압축 필수")은 cascade checkpointing 이 **없는** 상태에서 측정된 것이다.
#     실제로 promptmr_plus.py 의 cascade 루프는 checkpoint 없이 생으로 돌고 있었고,
#     sens 경로는 코일별로 쪼개 계산하면서도 바깥에서 통짜 reentrant checkpoint 를 걸어
#     역전파 재계산 때 15코일분 activation 이 한꺼번에 살아났다 — 즉 코일별 분할의 절감 효과가
#     사실상 무효였다. 둘 다 고쳤으므로 "압축 없이는 불가능"이라는 전제를 다시 재야 한다.
#
#   최악 shape 기준: readout-crop 높이 384 × 학습셋 최대폭 W=480 × 물리코일 15.
#     (W=480 은 train 170개 중 1개뿐이지만, 이 1개가 OOM 상한을 혼자 결정한다.)
#
#   ⚠️ 탐침은 config 당 프로세스 1개로 돌린다(VRAM 을 깨끗이 초기화하기 위해).
#   ⚠️ OOM 은 이 슬롯의 정상 출력이다. 스크립트가 죽지 않고 다음 config 로 넘어간다.
#
# 실행:  bash shell13.sh   (GPU 비었을 때)
# ============================================================
set -uo pipefail
export PYTHONUNBUFFERED=1

# --- repo 루트로 이동 ---
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.." || exit 1

# --- 로그 파일 ---
SLOT="$(basename "$0" .sh)"
TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="shell_scripting/logs"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/${SLOT}_${TS}.log"
exec > >(tee -a "$LOG") 2>&1

echo "=========================================================="
echo "[START] $SLOT   $(date '+%F %T')"
echo "[CWD  ] $(pwd)"
echo "[LOG  ] $LOG"
echo "=========================================================="

# --- 가드: GPU 비었는지 (used < 500MiB) ---
USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
echo "[GPU ] used=${USED}MiB"
if [ -n "$USED" ] && [ "$USED" -ge 500 ]; then
    echo "[STOP] GPU가 사용 중(${USED}MiB)이다. 탐침은 GPU가 빈 상태라야 peak 이 의미가 있다."
    exit 2
fi

H=384      # readout-crop 적용 시 높이
W=480      # 학습셋 최대 폭 = 최악 케이스
PROBE="python scripts/probe_oom_promptmr.py --crop-h $H --width $W --n-feat0 48 --sens-n-feat0 24 --n-history 3"

echo ""
echo "=== A) 기준선: 압축했다고 가정한 코일 8개 (기존 확정 config nc=4) ==="
$PROBE --coils 8 --num-cascades 4

echo ""
echo "=== B) 코일 전부(15). checkpointing 유무 비교 — 이번 수정의 효과 ==="
$PROBE --coils 15 --num-cascades 4 --no-use-checkpoint
$PROBE --coils 15 --num-cascades 4

echo ""
echo "=== C) 코일 전부(15) + cascade 스윕 — nc 를 어디까지 올릴 수 있나 ==="
for NC in 6 8 10 12; do
    $PROBE --coils 15 --num-cascades $NC
done

echo ""
echo "=========================================================="
echo "[DONE ] $SLOT   $(date '+%F %T')"
echo "[읽는 법]"
echo "  - B 의 두 줄 차이 = cascade checkpointing + 코일별 sens checkpointing 이 실제로 번 양."
echo "  - C 에서 OK 로 끝나는 가장 큰 nc 가 코일 전부 쓰면서 감당 가능한 cascade 수."
echo "    peak 이 ~7.0GB 를 넘으면 실학습에선 쓰지 말 것(폭 변동·단편화 여유 필요)."
echo "  - 전부 OOM 이면 코일 압축을 되살리는 게 아니라 먼저 원인을 다시 볼 것:"
echo "    checkpointing 이 정말 켜졌는지(--use-checkpoint), compute_sens_per_coil 이 True 인지."
echo "[다음] 통과한 nc 를 shell14.sh 의 --num-cascades 에 넣고 실학습."
echo "=========================================================="
