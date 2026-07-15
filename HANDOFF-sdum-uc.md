# HANDOFF — SDUM Universal Conditioning (`--uc`) → GPU에서 이어서 검증

> **읽는 대상:** GPU 환경의 Claude CLI. 이 파일은 임시 인계 노트다. **다 읽고 검증 끝나면 삭제할 것.**
> **작성:** 2026-07-15, 로컬(Windows, torch 미설치) 세션에서. 브랜치 `SDUM`.
> **한 줄:** SDUM에 acc4/acc8 + cascade-index Universal Conditioning을 `--uc` 플래그 하나로 추가했다. 로컬엔 torch가 없어 **정적 검증(py_compile)까지만** 됐고 **런타임 검증은 네(GPU)가 해야 한다.**

---

## 0. 지금 당장 할 일 (요약)

1. `git pull` 로 SDUM 최신(이 커밋들 포함) 받기 — 이미 push됨.
2. **런타임 검증** (아래 §5) — pytest 5개 + smoke run + A/B. 이게 이 인계의 핵심 목적.
3. 검증 통과하면 이 파일(`HANDOFF-sdum-uc.md`) 삭제하고 커밋.
4. 실패하면 §4 고려사항 보고 고친 뒤 다시 검증.

---

## 1. 무엇을 만들었나

SDUM(vendored deep-unrolled MRI recon, `utils/model/sdum/`)에 **Universal Conditioning(UC)** 을 켜는 `--uc` 플래그 추가. UC = 논문 §3.5의 조건화로, cascade마다 **(a) cascade index(timestep)** 와 **(b) protocol label(여기선 acc4/acc8)** 을 sinusoidal 임베딩→MLP→Restormer의 GDFN에 bias로 주입한다.

- **기본 off.** `--uc` 없으면 `time_cond=label_cond=False` → 조건화 파라미터 자체가 안 생김 → **baseline과 state_dict 바이트 동일.** 기존 학습·체크포인트 100% 보존.
- acc 값은 **마스크 샘플링 비율 휴리스틱**(`_extract_acc`, `frac≤0.25→acc8`)으로 얻는다. 데이터 파이프라인은 안 건드렸다.
- `time_cond`가 `label_cond`의 전제다(`emb_channels`가 time_cond에만 물림) → `--uc`는 **항상 둘을 동시에** 켠다.

### 관련 커밋 (이 브랜치, origin/SDUM 위에 rebase됨)
- `docs(plan)` — 구현 계획 (`docs/superpowers/plans/2026-07-15-sdum-uc-acc-conditioning.md`, **자세한 근거·태스크는 여기**)
- `feat(sdum): tiny_8gb_args uc flag` — config factory에 `uc` 인자
- `feat(sdum): adapter wires timestep=0` — adapter가 UC 시 `timestep=0` 전달 + 3-튜플 리턴 처리
- `feat: --uc flag wires UC through train/infer/resume`
- `fix(sdum): resume back-compat, --uc in reconstruct.py, restore comment`

### 바뀐 파일
- `utils/model/sdum/config.py` — `tiny_8gb_args(uc=False, **overrides)`; `uc=True`면 time_cond+label_cond 켬
- `utils/model/sdum/adapter.py` — `self.uc` 저장; UC면 `self.cascaded(..., timestep=0)` 후 `out=out[0]` (3-튜플 언팩), 아니면 기존 텐서 경로
- `train.py`, `utils/learning/train_part.py`, `utils/learning/test_part.py`, `scripts/reconstruct.py` — `--uc` 배선
- `scripts/test_sdum_uc.py` — UC 단위 테스트 5개 (신규)

---

## 2. Git 상태 / 팀원 작업과의 병합 (중요)

내가 작업하는 동안 팀원이 origin/SDUM에 **C1(bbox 가중 손실) 커밋 6개**를 push했다(`--bbox-loss-lambda`, bbox_ssim_loss 등). 그래서 두 갈래가 diverge했고, **내 UC 커밋들을 origin/SDUM 위로 rebase해서 push**했다.

- **병합 충돌은 `train.py` 딱 하나였다:** 팀원의 `--bbox-loss-lambda`와 내 `--uc`가 둘 다 `--fg-loss` 바로 뒤에 argparse 라인을 추가 → 인접 충돌. **둘 다 남기는 것으로 해결**(둘 다 독립 플래그).
- `train_part.py`는 팀원(`train_epoch` 언팩·loss)과 나(`check_resume_compat`·`train()` 빌드)가 **다른 함수**라 자동 병합됨.
- **두 기능은 직교하고 둘 다 default-off다.** `--uc`와 `--bbox-loss-lambda`를 **동시에** 켜도 문제없어야 한다: bbox는 `train_epoch`에서 model 호출 **전에** data 튜플에서 떼어내 손실에만 쓰이고(모델 입력 아님), UC는 model 내부 조건화라 서로 안 닿는다. 단 **동시 사용은 아직 아무도 런타임 검증 안 함** — 필요하면 확인.

---

## 3. UC가 실제로 살아있는지 (구조 확인됨, 런타임 미확인)

경로(코드 정독 + opus 리뷰로 trace 완료, 하지만 **실행은 안 됨**):
`adapter.forward`(uc on)→`Cascaded_SkipConnected_MRI_Recon.forward(..., timestep=0)`→cascade마다 `actual_timestep=0*T+i=i`→`restormer_mri.forward`가 mask_type·acc·acq로 label 계산→`Restormer.forward`가 `emb=time_embed(t)+label_emb(y)`→`GDFN`이 `emb_layers(emb)`를 bias로 더함.

- 현 설정 `n_mask=1, n_acq=1, n_acc=2` → `label=acc_idx∈{0,1}` 로 acc4/acc8 깔끔히 분리.
- **과거 함정:** `time_cond`만 켜면 adapter가 timestep을 안 넘겨 조건화가 **조용히 무동작**(에러도 안 남)이었다. 이번에 adapter가 `timestep=0`을 넘기게 고쳐서 해결. `scripts/test_sdum_uc.py::test_uc_conditioning_is_actually_used`가 **UC 파라미터에 grad가 실제로 흐르는지**로 이 무동작을 잡는다 — 이 테스트가 GPU에서 통과하는지 꼭 확인.

---

## 4. 고려할 사항 (검증·후속 작업 시)

1. **⚠️ UNVERIFIED.** 로컬 torch 없음 → 전부 `python -m py_compile`까지만. **pytest·smoke·학습은 한 번도 안 돌았다.** 런타임 버그 가능성 상존 — §5 필수.
2. **⚠️ 마스크 증강과 UC 동시 사용 금지.** `_aug_mask`가 주변부 라인을 솎아 frac을 낮춤 → acc4가 acc8로 오판된다. UC 실험은 마스크 증강 **off**에서만(현재 `aug_on` 기본 off, `--aug-*` 인자도 없어 안전).
3. **resume 호환:** `check_resume_compat`에 `uc`를 넣되 **옛 체크포인트(uc 필드 없음)=UC off로 간주**(하위호환). UC on↔off resume은 아키텍처가 달라 **명시적 ValueError로 거부**된다. UC 학습은 **새 `-n <net_name>` 으로 fresh run** 할 것(옛 baseline 체크포인트에서 `--resume` 불가).
4. **config footgun (그대로 두기로 결정함, 손대지 말 것):** `tiny_8gb_args(uc=True, time_cond=False)`는 `label_cond=True/time_cond=False`라는 빌드 불가 조합을 만들 수 있다(모델 생성 시 `nn.Linear(C,None)` TypeError). `--uc`는 항상 둘 다 켜므로 실제 caller는 없음 → YAGNI로 방치 확정. `test_uc_explicit_override_wins`가 이 조합을 args 수준에서만 검사(빌드 안 함)라 지금은 안전.
5. **batch=1 전제:** adapter가 배치 0번 마스크로 acc를 정하고 label을 배치 전체에 broadcast. batch>1에서 acc가 섞이면 틀림. 배치 키울 계획 없어 가드 안 넣음(`train.py` 기본 `-b 1`).
6. **acc 판별 테스트는 없음:** grad-flow 테스트는 "조건화가 쓰임"을 증명하지만 "acc4 vs acc8 출력이 실제로 다름"은 아니다. label이 마스크에서 파생돼 같은 마스크로 다른 acc를 못 줘서, adapter 레벨에선 깔끔히 격리 불가(순진한 "다른 마스크→다른 출력"은 UC off여도 통과하는 vacuous 테스트). 필요하면 cascade를 직접 호출하는 통합 테스트로 별도 작성.

---

## 5. 런타임 검증 런북 (GPU에서 실행 — 이 인계의 목적)

데이터 경로는 실제 값으로 바꿀 것. (계획서 Task 3·4에 동일 내용 있음.)

```bash
# 1) UC 단위 테스트 5개 (빌드 / forward shape / off 보존 / grad 흐름=무동작 방지)
pytest scripts/test_sdum_uc.py -v
#   특히 test_uc_conditioning_is_actually_used 통과가 핵심.

# 2) smoke run: 크래시 없이 학습되고 체크포인트에 uc=True가 저장되는지
python train.py --model sdum --uc --num-cascades 2 --channels 16 32 \
  -e 1 --max-iters 3 -n uc_smoke -t /Data/train/ -v /Data/val/
python -c "import torch; a=torch.load('../result/uc_smoke/checkpoints/best_model.pt', weights_only=False)['args']; print('uc=', getattr(a,'uc',None))"
#   기대: uc= True

# 3) resume 비호환이 명시적 에러를 내는지 (uc off 체크포인트를 uc on으로 재개 시도)
python train.py --model sdum          --num-cascades 2 --channels 16 32 -e 1 --max-iters 3 -n uc_offsmoke -t /Data/train/ -v /Data/val/
python train.py --model sdum --uc --resume --num-cascades 2 --channels 16 32 -e 1 --max-iters 3 -n uc_offsmoke -t /Data/train/ -v /Data/val/
#   기대: 두 번째가 ValueError: --resume 불일치: 'uc' 저장=False 현재=True 로 중단.

# 4) A/B 실효성 측정 (계획서 Task 4): 같은 config로 off vs on 학습 후 val 비교
python train.py --model sdum       --num-cascades 4 --channels 32 64 -e 5 -n uc_off -t /Data/train/ -v /Data/val/
python train.py --model sdum --uc  --num-cascades 4 --channels 32 64 -e 5 -n uc_on  -t /Data/train/ -v /Data/val/
python -c "
import numpy as np
off=np.load('../result/uc_off/val_loss_log_.npy'); on=np.load('../result/uc_on/val_loss_log_.npy')
print('off best(1-SSIM):', off[:,1].min(), ' on:', on[:,1].min(), ' delta(+면 UC개선):', off[:,1].min()-on[:,1].min())
"
```

**해석 주의:** 논문 Table 3f는 UC +0.38dB지만 그건 11 contrast × 다중 trajectory의 universal 세팅. 여기 knee는 조건 신호가 **acc4/acc8 1비트뿐**이라 이득이 그보다 작을 수 있다. off/on A/B로 실제로 val이 오르는지 보고 UC 유지 여부를 판단할 것.

---

## 6. 포인터
- 계획서(근거·태스크·self-review): `docs/superpowers/plans/2026-07-15-sdum-uc-acc-conditioning.md`
- UC 조건화 원리 정리(FiLM vs UC): `../papers/study/sdum-film-conditioning.md` (repo 밖, 로컬 전용일 수 있음)
- ⚠️ `.superpowers/sdd/`(진행 원장·리뷰 기록)는 **git-ignore라 여기 없다.** 이 파일이 그 내용의 자기완결 요약이다.

**검증 끝나면 이 파일 삭제할 것.**
