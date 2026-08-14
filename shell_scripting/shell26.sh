#!/usr/bin/env bash
# ============================================================
# 목적: MoE(acc4/acc8 분리)가 이득인지 층화 k-fold hold-out에서 판정한다        [shell26.sh]
# run tag : promptmr_moe_f<FOLD>_{shared,acc4,acc8} → 병합본 promptmr_moe_f<FOLD>
#
#   ① pretrain  --acc all            → 공유 모델                (= 단일 모델 baseline 겸용)
#   ② ft4       --acc 4 --init-from ① → acc4 전문가
#   ③ ft8       --acc 8 --init-from ① → acc8 전문가
#   ④ merge     ②+③ → 결합 체크포인트 (추론 시 마스크로 자동 라우팅)
#   ⑤ score     leaderboard 가 있으면 recon_eval, 없으면 건너뜀
#
#   [이 슬롯이 shell20과 다른 점] shell20은 고정 분할(-t train/ -v val/)이었다. 여기는
#     층화 k-fold를 쓴다 — 세 단계가 **같은 hold-out**을 본다는 게 이 실험의 전제다.
#     층화(2026-07-26) 이전 코드였다면 --acc 를 줄 때마다 fold 경계가 움직여서
#     ②③이 ①의 hold-out 볼륨으로 학습했다(실측 40개 중 30개). 지금은 안 그렇다.
#
#   [판정 방법] 셋 다 같은 hold-out 위에 있으므로 val 을 나란히 읽을 수 있다.
#     ①의 val 은 fold 전체(acc4+acc8) 평균이라 ②③(각 acc 절반)과 직접 비교되지 않는다.
#     → **판정의 정본은 채점이다.** recon_eval 이 acc4/acc8 을 따로 찍어주므로
#        ①(단일)과 ④(MoE)를 각각 채점해 SSIM_full·SSIM_bbox·시간 3축으로 비교한다.
#        val 은 학습이 살아있는지 보는 계기판으로만 쓴다.
#
#   [미세조정 하이퍼] lr 은 pretrain 과 같은 2e-4, 5 epoch. shell20의 5e-5×3ep 은
#     3 epoch 내내 소수 4째 자리만 흔들려(0.07694→0.07690→0.07694) 미세조정을 했다고
#     말할 수 없는 상태였다. lr 이 낮았던 건지 이미 수렴인지 가르려고 lr 을 올렸다.
#
#   [증강] 기본 ON (AUG=0 으로 끔). 좌우 flip 하나뿐이다 — mask thinning 은 2026-07-28 코드에서
#     제거됐다(대회 입력이 acc4/acc8 고정이라 없는 가속률을 학습하고, MoE 라우터 기준으로는
#     acc4 샘플이 acc8 쪽 입력으로 바뀌어 이 슬롯의 전문가 분리 판정까지 흐렸다).
#     ⚠️ 증강 이득은 아직 A/B 로 확인된 적이 없다. 그래도 켜는 건 남은 시간에 "MoE + 증강"을
#        한꺼번에 노리는 선택이고, 대신 **결과가 나빠도 둘 중 뭐 때문인지 이 런만으론 못 가린다.**
#        증강을 격리하려면 AUG=0 으로 pretrain 을 한 번 더 돌려 ①끼리 비교해야 한다.
#     스케줄은 constant 0.5 · delay 0 — 첫 epoch부터 매 샘플 절반이 뒤집힌다(2026-07-29).
#
#   [마스크 재생성] REMASK=1 로 켠다 (기본 0). ②③ 의 --acc 4|8 을 --acc all --remask 4|8 로 바꾼다.
#     학습풀 h5 의 kspace 는 마스크 곱 이전 full k-space 라(train 170/170·val 30/30 실측),
#     딸려온 마스크를 버리고 대회 생성기로 만든 마스크를 곱하면 **같은 볼륨을 반대 가속률로** 쓸 수 있다.
#     정답 image_label 은 fully-sampled 복원이라 마스크를 바꿔도 그대로 유효하다.
#     → acc4 전문가는 acc8 파일에서 뽑은 acc4 를, acc8 전문가는 acc4 파일에서 뽑은 acc8 을 함께 본다.
#        전문가당 학습 볼륨 80 → 160 (2195/2163 → 4358 slice, 정확히 2배).
#     생성기는 학습풀 200 + leaderboard 80 전수에서 center_fraction 후보 완전탐색 결과
#     네 셀 모두 cf=0.08 만 유일하게 비트정확 재현(실패 0) — 우리 마스크와 채점 마스크가 같은 물건이다.
#     offset 은 기본 uniform: 학습풀은 offset==(W//2)%acc 로 결정론인데 **채점은 무작위 0~acc-1** 이라,
#     지금 acc8 전문가는 채점 볼륨의 65% 를 학습에서 본 적 없는 offset 으로 만난다(REMASK_OFFSET=canonical 로
#     분포를 안 건드리고 볼륨 풀만 2배로 늘릴 수도 있다).
#     ⚠️ val 은 그대로다 — 딸려온 마스크 + 그 전문가의 acc 볼륨(과거 ft4/ft8 val 곡선과 비교 가능).
#     ⚠️ ① 은 안 건드린다 → NET_SHARED 에 접미사가 안 붙어 **기존 21h 체크포인트를 그대로 재사용**한다.
#        ①이 이 이름으로 없더라도 INIT_NET 으로 다른 슬롯 결과를 가리키면 된다 — shell22 의
#        promptmr_kfold_f<FOLD>_aug 는 이 슬롯 ①과 train.py 인자가 사실상 같다(--num-workers 만 다름).
#        init_compat_guard 가 그 체크포인트의 k-fold/fold/seed/acc 를 대조해 hold-out 이 어긋나면 막는다
#        (--init-from 은 check_resume_compat 을 안 타므로 이게 유일한 방어선이다).
#     ⚠️ ②③ 이 각 5.3h → 10.3h 가 된다(데이터 2배). 상수 LR·epoch 고정이라 optimizer 스텝도 2배라서,
#        이득이 나와도 "마스크 다양성" 인지 "스텝 2배" 인지 이 런만으로는 못 가린다.
#        스텝을 지금 수준으로 고정하려면 FT_CAP=<그 fold train slice 의 절반> (fold2 는 2179).
#
#   ⚠️ 실행 전 서버에서 층화 테스트를 먼저 통과시킬 것 (31시간 태우고 분할이 틀리면 손해가 크다):
#        python -m pytest scripts/tests/test_kfold_stratify.py scripts/tests/test_remask.py -v
#
# 실행:
#   SMOKE=1 STAGE=all bash shell_scripting/shell26.sh   # 각 단계 120 iter (먼저 이것)
#   STAGE=pretrain bash shell_scripting/shell26.sh      # ① 만 (~21h)
#   STAGE=ft4      bash shell_scripting/shell26.sh      # ② 만 (~5h, ① 필요
#   STAGE=ft8      bash shell_scripting/shell26.sh      # ③ 만 (~5h, ① 필요)
#   STAGE=merge    bash shell_scripting/shell26.sh      # ④+⑤
#   STAGE=all      bash shell_scripting/shell26.sh      # ①→⑤ 순차 (기본값, ~31h)
#   FOLD=3 STAGE=all bash ...                           # 다른 fold
#   AUG=0 STAGE=pretrain bash ...                       # 증강 끄고 (증강 격리 비교용)
#   REMASK=1 STAGE=ft4 bash ...                         # 마스크 재생성 ON (~10.3h, ① 재사용)
#   REMASK=1 REMASK_OFFSET=canonical STAGE=ft8 bash ... # 분포 그대로 두고 볼륨 풀만 2배
#   REMASK=1 FT_CAP=2179 STAGE=ft4 bash ...             # 스텝을 현행에 고정 (fold2 기준, ~5.3h)
#   INIT_NET=promptmr_kfold_f2_aug REMASK=1 STAGE=ft8 bash ...   # ①을 다른 슬롯 결과로 대체(21h 절약)
#   FORCE=1 ...                                         # 기존 결과를 덮어쓸 때만
# ============================================================
set -uo pipefail
export PYTHONUNBUFFERED=1

# --- repo 루트로 이동 ---
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.." || exit 1

# ===== [고정 인자] 런마다 안 바뀌는 것 — 여기 바꾸면 이후 모든 런의 baseline이 바뀐다 =====
#   ⚠️ 세 단계가 **완전히 같은 아키텍처**여야 한다. --init-from 이 가중치를 로드하고
#      merge_moe.py 가 두 전문가의 config 일치를 강제하므로(MUST_MATCH), 여기서 갈리면 병합이 거부된다.
# 데이터 루트 — 서버마다 다르다(vessl=/root/Data, runpod=/workspace/Data). 기본은 이 서버(vessl).
DATA_ROOT="${DATA_ROOT:-/root/Data}"
BASE_ARGS=( --model promptmr --readout-crop
    --num-cascades 10 --n-feat0 48 --sens-n-feat0 24 --n-history 3
    -b 1 -l 2e-4 -r 100 --num-workers 4 --use-checkpoint
    --fg-loss --area-weight --grad-clip 0.1 --accum-steps 2
    -t "$DATA_ROOT/train/" -v "$DATA_ROOT/val/" )

# ===== [이번 run] 여기만 채운다 =====
FOLD="${FOLD:-2}"
K_FOLD="${K_FOLD:-5}"
KFOLD_SEED="${KFOLD_SEED:-430}"
STAGE="${STAGE:-all}"
SMOKE="${SMOKE:-0}"

PRE_EPOCHS=10        # 공유 사전학습
FT_EPOCHS=5          # 미세조정 (lr 은 BASE_ARGS 의 2e-4 를 그대로 물려받는다)

# --- 증강(좌우 flip): 기본 ON. AUG=0 으로 끌 수 있다 ---
#   ⚠️ 증강 이득은 아직 A/B 로 확인 안 됐다(shell22 는 OFF arm 이 구파티션이라 무효). 그래도 켜는 건
#      "MoE 이득 + 증강 이득"을 한 번에 노리는 선택이다 — 시간이 없어 변수 분리를 포기한 것이니,
#      결과가 나쁘면 둘 중 뭐 때문인지 이 런만으로는 못 가린다는 걸 알고 읽을 것.
#      (증강만 격리하려면 AUG=0 으로 한 번 더 돌려 pretrain 끼리 비교해야 한다.)
#   [스케줄] constant 0.5 · delay 0 — 첫 epoch부터 끝까지 매 샘플을 절반 확률로 뒤집는다.
#     exp+delay 램프는 MRAugment 가 회전·스케일·shear 의 보간 아티팩트 때문에 두는 장치다.
#     우리한테 남은 flip 은 closed form 이라 픽셀 정확하고 결과도 물리적으로 유효한
#     (반대쪽 무릎) 획득이라 미룰 이유가 없다. 램프를 쓰면 런 평균 노출이 28% 로 깎인다
#     (exp·D=3·T=10 을 epoch 에 대해 적분한 값). constant 0.5 는 같은 시간에 50%.
AUG="${AUG:-1}"
AUG_COMMON=( --aug-on --aug-schedule constant --aug-max-prob 0.5 --aug-delay 0 )

# --- 마스크 재생성: 기본 OFF. REMASK=1 로 켠다 (②③ 에만 적용, ① pretrain 은 불변) ---
REMASK="${REMASK:-0}"
REMASK_OFFSET="${REMASK_OFFSET:-uniform}"

LEADERBOARD="${LEADERBOARD:-/root/Data/leaderboard}"

# --- net_name: fold 를 이름에 넣는다. 안 넣으면 옛 promptmr_moe_shared(고정분할)와 섞인다 ---
#   ①(shared)에는 _rm 을 안 붙인다 — 재생성은 ②③ 만 바꾸므로 **기존 pretrain 체크포인트를 재사용**하고,
#   두 arm(_rm 유무)이 같은 초기값에서 갈라져야 비교가 성립한다.
SUF=""
[ "${AUG:-1}" = "1" ] && SUF="${SUF}_aug"     # 증강 런은 이름을 달리한다(비증강 런과 섞이면 비교가 깨진다)
FTSUF="$SUF"
[ "$REMASK" = "1" ] && FTSUF="${FTSUF}_rm"
[ "$SMOKE" = "1" ] && { SUF="${SUF}_smoke"; FTSUF="${FTSUF}_smoke"; }
NET_SHARED="promptmr_moe_f${FOLD}_shared${SUF}"
NET_ACC4="promptmr_moe_f${FOLD}_acc4${FTSUF}"
NET_ACC8="promptmr_moe_f${FOLD}_acc8${FTSUF}"
NET_MOE="promptmr_moe_f${FOLD}${FTSUF}"

# --- ① 재사용: 다른 슬롯이 만든 pretrain 을 그대로 초기값으로 쓴다 (21h 절약) ---
#   shell22(promptmr_kfold_f<FOLD>_aug) 는 이 슬롯의 ①과 train.py 인자가 사실상 같다 —
#   아키텍처·k-fold/fold/seed·--acc all(기본값)·10 epoch·lr·accum·grad-clip·fg-loss·
#   area-weight·증강(constant 0.5·delay 0) 전부 동일하고, --num-workers(6 vs 4)만 다른데
#   그건 dataloader 병렬도라 학습 궤적에 영향이 없다(flip 은 (seed,epoch,fname,slice) crc32,
#   셔플은 seed+epoch generator → 워커 수와 무관).
#   ⚠️ INIT_NET 은 **읽기 전용 소스**다. ①의 출력 이름(NET_SHARED)은 안 바뀌므로
#      남의 결과 폴더에 덮어쓸 일이 없다. 대신 do_pretrain 이 INIT_NET 지정 시 거부한다.
#   예: INIT_NET=promptmr_kfold_f2_aug REMASK=1 STAGE=ft8 bash shell_scripting/shell26.sh
INIT_NET="${INIT_NET:-$NET_SHARED}"
SHARED_BEST="../result/${INIT_NET}/checkpoints/best_model.pt"

# --max-iters 는 여기 한 곳에서만 정한다 — 같은 명령에 두 번 붙이면 argparse 가 뒤엣것을 이겨
#   SMOKE 가 120 이 아니라 FT_CAP 으로 돌아버린다. pretrain 은 EXTRA, 미세조정은 FT_EXTRA.
if [ "$SMOKE" = "1" ]; then
    EXTRA="--max-iters 120"; PRE_EPOCHS=1; FT_EPOCHS=1
    FT_EXTRA="$EXTRA"
else
    EXTRA=""
    FT_EXTRA=""
    # FT_CAP: 데이터는 2배로 늘리되 optimizer 스텝은 현행에 고정하는 arm(미세조정에만 적용).
    #   매 epoch 셔플된 4358 중 다른 절반을 뽑으므로 5 epoch 누적 커버리지 1−(1/2)^5 ≈ 96.9%.
    #   ⚠️ 정수 검증을 먼저 한다. `[ "$FT_CAP" -gt 0 ]` 만 쓰면 "2,179" 같은 오타에서 bash 가
    #      integer expression 오류를 뱉고 -e 가 없어 그냥 진행 → 캡이 조용히 사라진 채
    #      10.3h 짜리 무캡 런이 돈다. 게다가 그 오류는 로그 리다이렉트(아래 exec) 전이라 로그에 안 남는다.
    case "${FT_CAP:-0}" in
        ''|*[!0-9]*) echo "[STOP] FT_CAP 은 0 이상의 정수여야 한다 (받은 값: '${FT_CAP}')"; exit 2 ;;
    esac
    [ "${FT_CAP:-0}" -gt 0 ] && FT_EXTRA="--max-iters ${FT_CAP}"
fi

# 증강 인자(AUG=0 이면 빈 배열 → 증강 off). 스케줄이 constant·delay 0 이라 단계별로 다를 게
#   없다 — 스모크(1 epoch)에서도 epoch 0 부터 p=0.5 라 "배선됐는지" 확인이 그대로 된다.
AUG_ARGS=()
[ "$AUG" = "1" ] && AUG_ARGS=( "${AUG_COMMON[@]}" )

# --- 로그 ---
SLOT="$(basename "$0" .sh)"
TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="shell_scripting/logs"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/${SLOT}_f${FOLD}_${STAGE}_${TS}.log"
exec > >(tee -a "$LOG") 2>&1

echo "=========================================================="
echo "[START] $SLOT (fold $FOLD / $K_FOLD)  STAGE=$STAGE  SMOKE=$SMOKE  $(date '+%F %T')"
echo "[CWD  ] $(pwd)"
echo "[LOG  ] $LOG"
echo "[BR   ] $(git branch --show-current 2>/dev/null) ($(git rev-parse --short HEAD 2>/dev/null))"
echo "[CONF ] nc10 · nf48 · sens24 · lr 2e-4 · seed=$KFOLD_SEED · pretrain ${PRE_EPOCHS}ep · ft ${FT_EPOCHS}ep"
if [ "$REMASK" = "1" ]; then
    echo "[REMASK] ON (offset=$REMASK_OFFSET) — ②③ 만: --acc all --remask 4|8, 전문가당 80→160 volume."
    echo "         ① 은 불변이다. ft 이름에 _rm 이 붙는다."
    echo "         ft 시간 ≈ 5.3h → 10.3h. FT_EXTRA=[${FT_EXTRA}] ${FT_EXTRA:+(스텝 고정 arm)}"
else
    echo "[REMASK] OFF (REMASK=1 로 켬) — ②③ 이 딸려온 마스크·acc 필터 그대로."
fi
# ①이 어느 체크포인트인지는 REMASK 와 무관하게 항상 찍는다 — INIT_NET 을 빠뜨린 채 돌리면
#   ②③ 이 "공유 사전학습 결과가 없다"로 멈추는데, 그때 원인을 로그에서 바로 읽으려면 이 줄이 필요하다.
if [ "$INIT_NET" != "$NET_SHARED" ]; then
    echo "[INIT ] ① 소스 = $INIT_NET (다른 슬롯 재사용 — 이 슬롯의 $NET_SHARED 는 안 쓴다)"
else
    echo "[INIT ] ① 소스 = $INIT_NET (이 슬롯 자체 pretrain. 남의 걸 쓰려면 INIT_NET=<net 이름>)"
fi
echo "=========================================================="

# --- 가드: 층화 코드가 있는가 (이 슬롯의 전제. 없으면 --acc 가 fold 경계를 움직인다) ---
grep -q "_stratified_folds" utils/data/load_data.py 2>/dev/null || {
    echo "[STOP] utils/data/load_data.py 에 _stratified_folds 가 없다 — 층화 이전 코드다."
    echo "       이 상태로 돌리면 ②③이 ①의 hold-out 으로 학습한다. git pull 후 재실행할 것."
    exit 2; }

# --- 가드: cv2 (--fg-loss 의 foreground mask 가 cv2 를 import) ---
python -c "import cv2" 2>/dev/null || { echo "[STOP] cv2 import 실패 — bash /root/restore_env.sh 후 재실행."; exit 2; }

gpu_guard() {
    local used
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
    echo "[GPU  ] used=${used}MiB"
    if [ -n "$used" ] && [ "$used" -ge 500 ]; then
        echo "[STOP] GPU가 사용 중(${used}MiB)이다. 빈 상태에서 실행할 것."; return 2
    fi
    return 0
}

# 중단된 학습이면 이어서, 처음이면 새로 — --resume 을 상시로 붙이면 첫 실행이
# FileNotFoundError 로 죽는다(load_resume_checkpoint). 그래서 체크포인트 유무로 정한다.
resume_flag() {   # $1=net_name → "--resume" 또는 ""
    [ "${FORCE:-0}" = "1" ] && return 0   # FORCE=1 은 "처음부터 다시" 라는 뜻이니 resume 을 붙이지 않는다
    if [ -e "../result/$1/checkpoints/model.pt" ] || [ -e "../result/$1/checkpoints/latest.pt" ]; then
        echo "--resume"
    fi
}

# 새로 시작하는데 결과가 이미 있으면(=resume 도 아닌데 덮어쓰기) 멈춘다.
overwrite_guard() {   # $1=net_name
    if [ -n "$(resume_flag "$1")" ] && [ "${FORCE:-0}" != "1" ]; then
        return 0   # resume 경로라 덮어쓰기가 아니다
    fi
    if [ -e "../result/$1/checkpoints/model.pt" ] && [ "${FORCE:-0}" != "1" ]; then
        echo "[STOP] ../result/$1/checkpoints/model.pt 가 이미 있다 — 덮어쓰기: FORCE=1"; return 2
    fi
    return 0
}

# ==========================================================
#  ↓↓↓  단계 함수 — BASE_ARGS 뒤가 각 단계의 델타 전부다  ↓↓↓
# ==========================================================
# --init-from 은 check_resume_compat 을 안 탄다 → fold 가 다른 체크포인트를 물려도 조용히 통과하고,
#   그러면 이 슬롯의 hold-out 볼륨을 이미 학습한 가중치로 미세조정하게 된다(로그만 봐선 못 잡는 누수).
#   다른 슬롯의 pretrain 을 재사용할 때 이 검사가 유일한 방어선이다.
init_compat_guard() {
    [ -f "$SHARED_BEST" ] || return 0    # 없으면 do_ft 의 [STOP] 이 먼저 잡는다
    python - "$SHARED_BEST" "$K_FOLD" "$FOLD" "$KFOLD_SEED" <<'PY'
import sys, torch
p, k, vf, sd = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
a = torch.load(p, map_location="cpu", weights_only=False)["args"]
g = lambda f, d=None: getattr(a, f, d)
# 분할(=hold-out)만 검사한다. 아키텍처가 어긋나면 --init-from 의 state_dict 로드가 알아서 죽는다.
want = {"k_fold": k, "val_fold": vf, "kfold_seed": sd, "acc": "all"}
bad = {f: (g(f), v) for f, v in want.items() if g(f) != v}
print(f"[init ] 소스 config: model={g('model')} nc={g('num_cascades')} nf={g('n_feat0')} "
      f"sens={g('sens_n_feat0')} nh={g('n_history')} readout_crop={g('readout_crop')} "
      f"coil_compress={g('coil_compress')} | k-fold={g('k_fold')} fold={g('val_fold')} "
      f"seed={g('kfold_seed')} acc={g('acc')} epochs={g('num_epochs')} aug_on={g('aug_on')}")
if bad:
    print("[STOP ] --init-from 소스의 분할이 이 슬롯과 다르다 → hold-out 이 샌다:")
    for f, (got, exp) in bad.items():
        print(f"          {f}: 소스={got!r} 이 슬롯={exp!r}")
    sys.exit(2)
print("[init ] 분할 일치 — 같은 hold-out 이다.")
PY
}

do_pretrain() {
    if [ "$INIT_NET" != "$NET_SHARED" ]; then
        echo "[STOP] INIT_NET=$INIT_NET 로 ①을 재사용하도록 지정해놓고 STAGE=pretrain 을 돌리려 한다."
        echo "       21시간을 헛되이 태우는 것이다 — STAGE=ft8 / ft4 / merge 만 돌려라."
        return 2
    fi
    gpu_guard || return 2
    overwrite_guard "$NET_SHARED" || return 2
    local R; R="$(resume_flag "$NET_SHARED")"
    echo "[① pretrain] $NET_SHARED  (--acc all, fold $FOLD hold-out) ${R:+[resume]}"
    echo "[① aug] ${AUG_ARGS[*]:-(off)}"
    set -x
    python train.py "${BASE_ARGS[@]}" \
        --k-fold "$K_FOLD" --val-fold "$FOLD" --kfold-seed "$KFOLD_SEED" \
        --acc all -e "$PRE_EPOCHS" $EXTRA $R "${AUG_ARGS[@]}" \
        -n "$NET_SHARED" --run-tag "pmr_moe_f${FOLD}_shared"
    { local st=$?; set +x; } 2>/dev/null
    echo "[① pretrain] exit=$st"
    return $st
}

do_ft() {   # $1=acc(4|8)  $2=net_name
    gpu_guard || return 2
    if [ ! -f "$SHARED_BEST" ]; then
        echo "[STOP] 공유 사전학습 결과가 없다: $SHARED_BEST"
        # 이름이 안 맞아서 못 찾는 건지, 아예 없는 건지를 사람이 바로 판단할 수 있게 후보를 보여준다.
        #   net 이름은 FOLD 와 AUG 로 갈린다(_aug 유무) — 다른 조합으로 돌린 pretrain 이 있으면 여기 뜬다.
        #   *_shared* 만 훑으면 다른 슬롯의 --acc all 10ep 런(promptmr_kfold_f<FOLD>_*)이 안 보인다.
        #   그게 INIT_NET 으로 쓸 수 있는 유일한 후보인 경우가 많으므로 같이 보여준다.
        local cands
        cands="$(ls -1d ../result/*_shared*/checkpoints/best_model.pt \
                        ../result/promptmr_kfold_f${FOLD}*/checkpoints/best_model.pt 2>/dev/null \
                 | grep -v _smoke)"   # 스모크(120 iter)는 후보가 아니다 — 분할은 맞아서 가드를 통과해버린다
        if [ -n "$cands" ]; then
            echo "       이 서버에 있는 ① 후보 체크포인트:"
            echo "$cands" | sed 's|^|         |'
            echo "       → 이름이 다르면 FOLD/AUG 를 맞추거나, 위 후보를 INIT_NET=<net 이름> 으로 지정할 것"
            echo "         (AUG=0 이면 _aug 가 안 붙는다. 분할이 어긋나면 init_compat_guard 가 막는다)."
        else
            echo "       이 서버엔 pretrain 체크포인트가 하나도 없다."
        fi
        echo "       → ①부터 돌리려면:  REMASK=$REMASK STAGE=all bash shell_scripting/$(basename "$0")"
        echo "         (①은 재생성 없이 ~21h, 이어서 ②③이 remask 로 돈다)"
        echo "       → 다른 슬롯의 pretrain 을 재사용하려면 INIT_NET=<net 이름> 을 주면 된다"
        echo "         (예: INIT_NET=promptmr_kfold_f${FOLD}_aug — shell22 의 ON arm)"
        return 2
    fi
    init_compat_guard || return 2
    overwrite_guard "$2" || return 2
    local R; R="$(resume_flag "$2")"
    # 마스크 재생성: --acc 는 "어떤 **파일**을 읽을지", --remask 는 "그 k-space 에 어떤 **마스크**를
    #   씌울지"라 축이 직교한다. 그래서 --acc all 로 볼륨 풀을 열고(160) 마스크만 이 전문가의
    #   가속률로 만든다. val 은 load_data 가 --remask 에서 파생해 acc$1 볼륨 + 딸려온 마스크로 되돌린다.
    local ACC_ARGS=( --acc "$1" ) RMNOTE=""
    if [ "$REMASK" = "1" ]; then
        ACC_ARGS=( --acc all --remask "$1" --remask-offset "$REMASK_OFFSET" )
        RMNOTE="  [remask: 160vol → acc$1 마스크 재생성 (offset=$REMASK_OFFSET)]"
    fi
    # --resume 과 --init-from 은 배타적이고 resume 이 우선이다(train.py). 재개면 init-from 은 무시된다.
    echo "[② ft$1] $2  (${ACC_ARGS[*]} --init-from shared) ${R:+[resume]}${RMNOTE}"
    echo "[② aug] ${AUG_ARGS[*]:-(off)}"
    set -x
    python train.py "${BASE_ARGS[@]}" \
        --k-fold "$K_FOLD" --val-fold "$FOLD" --kfold-seed "$KFOLD_SEED" \
        "${ACC_ARGS[@]}" -e "$FT_EPOCHS" $FT_EXTRA $R "${AUG_ARGS[@]}" \
        --init-from "$SHARED_BEST" \
        -n "$2" --run-tag "pmr_moe_f${FOLD}_acc$1"
    { local st=$?; set +x; } 2>/dev/null
    echo "[② ft$1] exit=$st"
    return $st
}

do_merge() {
    echo "[④ merge] $NET_ACC4 + $NET_ACC8 → $NET_MOE"
    # --threshold 를 안 준다 = merge_moe.py 의 DEFAULT_ACC_THRESHOLD(0.25) 사용.
    # 여기 숫자를 박아두면 나중에 기본값이 바뀔 때 조용히 어긋난다(옛 0.1875 사고 재발).
    set -x
    python utils/model/merge_moe.py --acc4-net "$NET_ACC4" --acc8-net "$NET_ACC8" --out-net "$NET_MOE"
    { local st=$?; set +x; } 2>/dev/null
    [ "$st" -ne 0 ] && { echo "[④ merge] 실패 exit=$st"; return $st; }

    if [ "$SMOKE" = "1" ]; then echo "[⑤ score] SMOKE — 채점 생략"; return 0; fi
    if [ ! -d "$LEADERBOARD/acc4" ] || [ ! -d "$LEADERBOARD/acc8" ]; then
        echo "[⑤ score] leaderboard 없음($LEADERBOARD) — 채점 건너뜀."
        echo "          채점은 데이터가 있는 환경에서: 아래 두 net 을 각각 돌려 3축 비교."
        echo "            python recon_eval.py -n $NET_MOE   -p <leaderboard>   # MoE"
        echo "            python recon_eval.py -n $INIT_NET  -p <leaderboard>   # 단일 baseline"
        return 0
    fi
    gpu_guard || return 2
    local OUT_DIR="shell_scripting/results"; mkdir -p "$OUT_DIR"
    echo "[⑤ score] recon_eval — MoE 와 단일 baseline 을 둘 다 채점해야 비교가 된다."
    set -x
    python -u recon_eval.py -n "$NET_MOE" -p "$LEADERBOARD" \
        | tee "$OUT_DIR/recon_eval_${NET_MOE}_${TS}.txt"
    { local st=${PIPESTATUS[0]}; set +x; } 2>/dev/null
    [ "$st" -ne 0 ] && { echo "[⑤ score] MoE 채점 실패 exit=$st"; return $st; }
    # 단일 baseline 은 **실제로 ②③의 초기값이 된 체크포인트**($INIT_NET)여야 한다.
    #   NET_SHARED 로 박아두면 INIT_NET 으로 남의 pretrain 을 빌려 쓴 런에서 없는 net 을 채점하려다 죽는다.
    #   이미 채점해둔 net 이면 shell_scripting/results/ 에 사본이 있으니 이 단계를 건너뛰어도 된다.
    set -x
    python -u recon_eval.py -n "$INIT_NET" -p "$LEADERBOARD" \
        | tee "$OUT_DIR/recon_eval_${INIT_NET}_${TS}.txt"
    { st=${PIPESTATUS[0]}; set +x; } 2>/dev/null
    echo "[⑤ score] exit=$st  점수 사본: $OUT_DIR/ (git add 해서 공유)"
    return $st
}

# ---------- STAGE 디스패치 ----------
case "$STAGE" in
    pretrain) do_pretrain; STATUS=$? ;;
    ft4)      do_ft 4 "$NET_ACC4"; STATUS=$? ;;
    ft8)      do_ft 8 "$NET_ACC8"; STATUS=$? ;;
    merge)    do_merge; STATUS=$? ;;
    all)      do_pretrain && do_ft 4 "$NET_ACC4" && do_ft 8 "$NET_ACC8" && do_merge
              STATUS=$? ;;
    *) echo "[STOP] 알 수 없는 STAGE='$STAGE' (pretrain|ft4|ft8|merge|all)"; exit 2 ;;
esac

echo "=========================================================="
echo "[DONE ] $SLOT (fold $FOLD)  STAGE=$STAGE  $(date '+%F %T')  exit=$STATUS"
if [ "$SMOKE" = "1" ]; then
    echo "[SMOKE] 확인할 것:"
    echo "        (0) ⚠️ 증강이 실제로 걸렸는가 — 로그에 '[aug] epoch=0 p_flip=0.5000' 처럼 p_flip>0 이 찍혀야 한다."
    echo "            (AUG=1 인데 p_flip=0.0000 이거나 [aug] 줄 자체가 없으면 배선이 안 된 것이다.)"
    echo "        (1) 세 단계 다 OOM 없이 exit 0 인가"
    echo "        (2) [K-Fold] 로그의 val 구성이 **세 단계에서 일관적인가** — 이게 층화의 서명이다."
    echo "            ① --acc all 의 val (acc4 A · acc8 B) 에 대해 ② 는 acc4 A, ③ 은 acc8 B 여야 한다."
    echo "            어긋나면 층화가 안 먹은 것이니 본 학습을 돌리지 말 것."
    echo "        (3) merge 가 config 불일치로 거부되지 않았는가"
    if [ "$REMASK" = "1" ]; then
        echo "        (4) ②③ 에 '[remask] 전제 검증 통과 — N volume' 과"
        echo "            '[remask] train 로더: accN 마스크 재생성' 두 줄이 찍혔는가."
        echo "            (전제 검증이 실패하면 학습 시작 전에 죽는다 — 그게 정상 동작이다.)"
        echo "        (5) ②③ 의 [K-Fold] **train** 이 160 vol (acc4 80 · acc8 80) 로 2배가 됐는가."
        echo "            **val** 은 그대로 20 vol (한쪽 acc 만) 이어야 한다 — 늘어났으면 배선이 틀린 것이다."
        echo "        (6) val 로더 쪽엔 '[remask] val/forward 로더: 재생성 안 함' 이 찍혀야 한다."
    fi
else
    echo "[저장] val loss: ../result/promptmr_moe_f${FOLD}_*/val_loss_log_pmr_moe_f${FOLD}_*.npy"
    echo "       병합본  : ../result/${NET_MOE}/checkpoints/best_model.pt"
    if [ "$REMASK" = "1" ]; then
        # ①(shared)에는 _rm 이 안 붙으므로, ④−① 델타에는 **전문가 분리 + 마스크 재생성 + 스텝 2배**가
        #   전부 섞여 있다. 이걸 "MoE 이득"이라고 읽으면 다음 슬롯 설계가 그 위에 쌓인다.
        echo "[판정] ⚠️ 이 런은 REMASK=1 이라 ${NET_MOE} 와 ${INIT_NET} 의 차이가"
        echo "       '전문가 분리 + 마스크 재생성 + (FT_CAP 없으면) 스텝 2배' **합산**이다. MoE 이득이 아니다."
        echo "       · MoE 이득만  : promptmr_moe_f${FOLD}${SUF}(REMASK=0 병합본) 대 ${INIT_NET}"
        echo "       · 재생성 이득만: promptmr_moe_f${FOLD}${SUF} 대 ${NET_MOE}"
        echo "       두 번째 비교에 쓸 promptmr_moe_f${FOLD}${SUF} 가 없으면 REMASK=0 으로 ②③④를 한 번 더 돌려야 한다(~10.6h)."
    else
        echo "[판정] 정본은 채점이다 — ${NET_MOE}(MoE) 와 ${INIT_NET}(단일) 를 각각 채점해"
        echo "       SSIM_full·SSIM_bbox·시간 3축으로 비교할 것. val 은 계기판일 뿐이다."
    fi
fi
echo "=========================================================="
exit "$STATUS"
