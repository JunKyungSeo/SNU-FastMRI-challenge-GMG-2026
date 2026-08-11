#!/usr/bin/env bash
# 최종 run(promptmrp_GMG_final_0807) 3단계 공통 인자. train_final_*.sh 가 source 한다.
# ⚠️ 세 단계가 같은 값이어야 --init-from 이 성립한다. 한 곳에서만 고친다.
RUN="promptmrp_GMG_final_0807"
DATA_ROOT="$([ -d /workspace/Data ] && echo /workspace/Data || echo /root/Data)"
SHARED_BEST="../result/${RUN}_shared/checkpoints/best_model.pt"

BASE_ARGS=( --model promptmr --readout-crop
    --num-cascades 12 --n-feat0 48 --sens-n-feat0 24 --n-history 11
    -b 1 -l 2e-4 -r 50 --num-workers 4 --use-checkpoint
    --fg-loss --area-weight --grad-clip 0.1 --accum-steps 2
    --k-fold 5 --val-fold 2 --kfold-seed 430
    --l1-weight 0.1
    -t "$DATA_ROOT/train/" -v "$DATA_ROOT/val/" )

# 증강은 단계마다 다르므로 _aug_<단계>.sh 가 AUG_ARGS 로 넣어준다.
# source 를 빼먹으면 set -u 가 "AUG_ARGS[@]: unbound variable" 로 즉사시킨다 → 증강 없이 도는 사고가 안 난다.

# $1=NET  $2...=이번 단계 델타
run_final() {
    local net="$1"; shift
    local slot ts log st
    slot="$(basename "$0" .sh)"; ts="$(date +%Y%m%d_%H%M%S)"
    log="shell_scripting/logs/${slot}_${ts}.log"; mkdir -p shell_scripting/logs
    exec > >(tee -a "$log") 2>&1

    echo "[START] $slot  $(date '+%F %T')  NET=$net"
    # set -u 에 기대지 않는다 — bash 4.4 부터 빈 배열의 "${a[@]}" 는 set -u 위반에서 빠졌고,
    # 서버(Linux bash 5.x)와 여기(macOS 3.2)의 동작이 같다는 근거가 없다. 증강이 조용히 빠진 채
    # 12일을 도는 사고만은 막아야 한다.
    [ -n "${AUG_ARGS+x}" ] || { echo "[STOP] AUG_ARGS 없음 — _aug_<단계>.sh 를 source 안 했다"; exit 2; }
    if [ -e "../result/$net/checkpoints/model.pt" ] && [ "${FORCE:-0}" != "1" ]; then
        echo "[STOP] ../result/$net/checkpoints/model.pt 있음 — 이어서: RESUME=1 / 덮어쓰기: FORCE=1"
        exit 2
    fi
    set -x
    python train.py "${BASE_ARGS[@]}" "${AUG_ARGS[@]}" "$@" -n "$net" --run-tag "$net" \
        ${RESUME:+--resume}
    { st=$?; set +x; } 2>/dev/null
    echo "[DONE ] $slot  $(date '+%F %T')  exit=$st  → ../result/$net/checkpoints/best_model.pt"
    return "$st"
}

# tail 런 전용. tail 은 val 이 학습에 들어가 best 판정이 거짓말이라 **마지막 epoch** 을 써야 하는데,
# 채점부(test_part·merge_moe)는 best_model.pt 만 읽는다. 그냥 cp 하면 model.pt['model'] 이 원본 θ 라
# EMA 가 조용히 버려진다 — 여태 점수가 전부 θ_EMA 기준이었으므로 다른 물건을 제출하게 된다.
promote_last_epoch() {   # $1=NET
    python3 - "../result/$1/checkpoints" <<'EOF'
import sys, torch
p = sys.argv[1]
c = torch.load(p + "/model.pt", map_location="cpu", weights_only=False)
assert c.get("ema"), "EMA 없이 돈 런이다 — model.pt 를 그대로 쓰면 된다"
c["model"], c["ema"] = c["ema"]["shadow"], None
torch.save(c, p + "/best_model.pt")
print("[tail] epoch %d 의 EMA 가중치를 best_model.pt 로 (기존 best 덮어씀)" % c["epoch"])
EOF
}
