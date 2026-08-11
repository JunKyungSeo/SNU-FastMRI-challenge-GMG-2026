#!/usr/bin/env bash
# ============================================================
# 목적: promptmrp_0801 — shared(12ep) → acc4 → acc8 전문가(각 8ep) 3단계 학습    [shell31.sh]
#
# 실행:  bash shell_scripting/shell31.sh                   # 세 단계 순차
#        MODEL=ACC4 bash shell_scripting/shell31.sh        # 한 단계만 (SHARED|ACC4|ACC8)
#        RESUME=1 MODEL=SHARED bash shell_scripting/shell31.sh   # 이어서
#        FORCE=1 ...                                       # 기존 결과 덮어쓰기
# ============================================================
set -uo pipefail
export PYTHONUNBUFFERED=1

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.." || exit 1

DATA_ROOT="$([ -d /workspace/Data ] && echo /workspace/Data || echo /root/Data)"
RUN="promptmrp_0801"
SHARED_BEST="../result/${RUN}_shared/checkpoints/best_model.pt"
SLOT="$(basename "$0" .sh)"
LOG_DIR="shell_scripting/logs"; mkdir -p "$LOG_DIR"
NOTIFY="$SCRIPT_DIR/notify.sh"

# ===== [고정 인자] 세 단계 공통 =====
BASE_ARGS=( --model promptmr --readout-crop
    --num-cascades 10 --n-feat0 48 --sens-n-feat0 24 --n-history 3
    -b 1 -r 50 --num-workers 4 --use-checkpoint
    --fg-loss --area-weight --grad-clip 0.1 --accum-steps 2
    --k-fold 5 --val-fold 2 --kfold-seed 430
    --aug-on --aug-schedule constant --aug-max-prob 0.5 --aug-delay 0
    --l1-weight "${L1_W:-0.1}"
    -t "$DATA_ROOT/train/" -v "$DATA_ROOT/val/" )

# 강제종료(Ctrl-C 등)면 종료 메일을 보내지 않는다
KILLED=0
trap 'KILLED=1' INT TERM

# epoch 요약 줄(TrainTime 이 있는 줄 — iter 로그와 구분됨)에서 1 epoch 소요와 끝낸 epoch 번호를
# 읽어 예상 종료 시각(KST)을 메일로 보낸다. resume 이면 남은 epoch 이 그만큼 줄어든다.
eta_watch() {   # $1=log  $2=stage  $3=총 epoch
    local out secs left eta
    while :; do
        out=$(awk '/TrainTime = /{d=$4; sub("/","",d); for(i=1;i<=NF;i++) if($i=="TrainTime"||$i=="ValTime") s+=$(i+2); print int(s), d+1; exit}' "$1" 2>/dev/null)
        [ -n "$out" ] && break
        sleep 60
    done
    secs="${out% *}"; left=$(( $3 - ${out#* } ))
    [ "$left" -gt 0 ] || return 0
    eta=$(TZ=Asia/Seoul date -d "+$((secs * left)) seconds" '+%F %T')
    bash "$NOTIFY" "$SLOT $2 예상 종료 $eta KST" \
        "1 epoch ≈ ${secs}s · 남은 $left epoch → 예상 종료 $eta KST"
}

run_stage() {   # $1=SHARED|ACC4|ACC8
    local stage="$1" net epochs acc st w ts log R=""
    local -a delta=()
    case "$stage" in
        SHARED)
            # patience 를 6 으로 두는 건 ep0~4 가 의도적 상수 LR 구간이라서다 — 3 이면 감쇠가
            # 시작되기도 전에 끊길 수 있다.
            net="${RUN}_shared"; epochs=12
            delta=( --acc all -e "$epochs" -l 2e-4
                    --lr-decay-start-epoch 5 --lr-min-factor 0.05 --mask-aug full
                    --early-stop-patience "${PATIENCE_SHARED:-6}" ) ;;
        ACC4|ACC8)
            acc="${stage#ACC}"; net="${RUN}_acc${acc}"; epochs=8
            delta=( --acc all --remask "$acc" -e "$epochs" -l 1e-4
                    --warmup-iters 200 --lr-decay-start-epoch 0 --lr-min-factor 0.05
                    --mask-aug offset --max-iters 2179 --init-from "$SHARED_BEST"
                    --early-stop-patience "${PATIENCE:-3}" ) ;;
    esac

    if [ "$stage" != SHARED ] && [ ! -f "$SHARED_BEST" ]; then
        echo "[STOP] 공유 모델이 없다: $SHARED_BEST  — MODEL=SHARED 부터 돌릴 것"; return 2
    fi
    if [ "${RESUME:-0}" = "1" ]; then
        [ -e "../result/$net/checkpoints/model.pt" ] || [ -e "../result/$net/checkpoints/latest.pt" ] || {
            echo "[STOP] $net 체크포인트가 없다 — RESUME=1 로 이어갈 게 없다"; return 2; }
        R="--resume"
    elif [ -e "../result/$net/checkpoints/model.pt" ] && [ "${FORCE:-0}" != "1" ]; then
        echo "[STOP] ../result/$net/checkpoints/model.pt 가 이미 있다 — 이어서: RESUME=1 / 덮어쓰기: FORCE=1"; return 2
    fi

    ts="$(date +%Y%m%d_%H%M%S)"; log="$LOG_DIR/${SLOT}_${stage}_${ts}.log"
    echo "=========================================================="
    echo "[START] $SLOT $stage   $(date '+%F %T')   NET=$net"
    echo "[LOG  ] $log"
    echo "=========================================================="
    eta_watch "$log" "$stage" "$epochs" & w=$!
    set -x
    python train.py "${BASE_ARGS[@]}" "${delta[@]}" $R \
        -n "$net" --run-tag "$net" 2>&1 | tee "$log"
    { st=${PIPESTATUS[0]}; set +x; } 2>/dev/null
    pkill -P "$w" 2>/dev/null; kill "$w" 2>/dev/null   # sleep 자식까지 같이(안 그러면 최대 60s 남는다)
    echo "[DONE ] $SLOT $stage   $(date '+%F %T')   exit=$st"
    [ "$KILLED" = "1" ] || bash "$NOTIFY" "$SLOT $stage 종료 (exit=$st)" \
        "$net · $(TZ=Asia/Seoul date '+%F %T') KST · $log"
    return "$st"
}

M="${MODEL:-ALL}"
case "$M" in
    SHARED|ACC4|ACC8) run_stage "$M"; STATUS=$? ;;
    ALL) run_stage SHARED && run_stage ACC4 && run_stage ACC8; STATUS=$? ;;
    *) echo "[STOP] MODEL 은 SHARED|ACC4|ACC8 (미지정=세 단계 순차)"; exit 2 ;;
esac

echo "[저장] 가중치: ../result/${RUN}_{shared,acc4,acc8}/checkpoints/best_model.pt"
echo "[다음] 채점: bash shell_scripting/shell31_recon.sh"
exit "$STATUS"
