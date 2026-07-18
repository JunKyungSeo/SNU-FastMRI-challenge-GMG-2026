# Universal condition analysis — 코드만 보고 SDUM 논문 정합성 점검 (2026-07-18)

> 이 문서는 run별 학습 분석이 아니라 **UC(Universal Conditioning) 코드가 SDUM 논문/공식 구현대로 됐는지**만 본다. run 로그·수치는 배경으로만 쓰고, 판정 근거는 전부 코드(파일:줄)다. 대조 대상 = 공식 참조 `NV-SDUM-official/`.

**한 줄 결론:** UC의 **모델 부품 코드는 공식 참조와 사실상 바이트 동일**(충실한 복제)이다. 문제는 코드가 틀린 게 아니라, **이 대회(knee 단일·uniform·acc4/acc8) 세팅에선 UC가 줄 수 있는 신호가 사실상 1비트뿐**인데다, repo가 **참조는 안 켜는 조건화 경로(cascade-index)까지 일부러 켰고**, 조건화가 **zero-init이 아니라 처음부터 노이즈성 bias를 주입**한다는 점이다. 즉 "구현은 맞지만 이 문제에 안 맞는(그리고 시간만 확실히 더 먹는) 레버"라는 게 코드가 말하는 결론이다.

---

## §0 핵심 요약 (판정)

- ✅ **모델 부품은 충실 복제 (F1, 확정).** repo `restormer.py`의 UC 부품(sinusoidal 임베딩·`time_embed`·`label_emb`·GDFN `emb_layers` bias 주입)은 공식 `restormer.py`와 **import 경로 2줄 빼고 전부 동일**. 조건화 로직 자체엔 손 안 댔다. → "논문 조건화를 잘못 베꼈다"는 아니다.
- ⚠️ **배선은 참조와 다르다 (F2, 확정 = faithfulness divergence).** 공식은 cascaded(비-flow) 모델을 부를 때 **timestep을 안 넘겨** cascade-index 조건화를 **꺼둔 채로** 돌린다. repo 어댑터는 `timestep=0`을 넘겨 그 경로를 **일부러 깨운다**. 참조가 이 index 조건화를 쓰는 건 오직 **flow 모드**인데, 우리는 `flow=False`다. 즉 **flow 전용 장치를 비-flow에 옮겨 붙인 것.**
- ⚠️ **UC가 줄 신호가 1비트로 붕괴 (F3, 확정 = 구조적 부적합).** knee는 mask=uniform 1종·acq=knee 1종·acc={4,8} 2종 → protocol label이 정확히 `acc_idx ∈ {0,1}`. 논문의 "11 contrast × 다중 궤적" universal 다양성이 **여기엔 존재하지 않는다.** UC 기계장치는 다 있는데 태울 연료가 acc4/acc8 한 비트뿐.
- ⚠️ **time_cond ↔ label_cond 강제 결합 (F4, 확정 = latent trap).** `emb_channels`가 `time_cond`에만 물려 있어, label만 켜면 `nn.Linear(C, None)`으로 **빌드 크래시**. 그래서 `--uc`는 **항상 둘 다** 켠다 → acc label만 쓰고 싶어도 **cascade-index MLP + 블록마다 emb_layers**가 딸려 온다 (= param↑·시간↑의 주범).
- ⚠️ **cascade-index 조건화는 사실상 중복 (F5, 확정).** cascade들은 **가중치 공유 안 함**(각자 다른 `restormer_mri` 인스턴스). 게다가 `flow=False`라 timestep은 항상 0 → index는 그냥 `i`(0~3) 상수. 각자 다른 가중치를 가진 모듈에 "너 몇 번째야"를 상수로 알려주는 건 자기 파라미터로 흡수 가능 = 얻는 것 거의 없음.
- ⚠️ **조건화가 zero-init이 아님 (F6, 확정 = 구조적 부적합).** `time_embed`·`label_emb`·`emb_layers` 전부 기본 init → **학습 0스텝부터** GDFN마다 `x = x + emb_out`으로 **0 아닌 랜덤 bias**를 주입. DiT/ADM은 조건화를 zero-init해서 "무조건화 baseline에서 출발"시키는데, 여긴 그게 없다. (단 이것도 **공식과 동일** — 공식은 대규모·장기 학습이라 이 워밍업 비용을 흡수함. 우리 10ep엔 순비용.)
- 🐛 **latent 인덱싱 버그 (F9, 확정이나 현재 무해).** label mixed-radix stride가 `len(mask_types)` 자리에 `len(acc_factors)`가 와야 맞다. 지금은 mask 1종이라 mask_idx=0이라 안 터짐. 공식도 똑같이 이렇게 돼 있음. mask family 추가하면 그때 버그.
- ⚠️ **추론 경로 footgun (완결성 비평, medium).** `reconstruct.py→test_part.forward()`는 모델을 **체크포인트가 아니라 CLI `--uc`로** 재구성한다. UC로 학습한 체크포인트를 `--uc` 없이 recon하면 `load_state_dict`가 **크래시**(strict=True). recon_eval이 쓰는 `load_model`은 체크포인트의 `saved.uc`로 자기기술적으로 복원하는데, 이쪽만 안 맞음. 조용한 오작동은 아니고 "터진다"라 그나마 다행.
- ℹ️ **조건화 형태가 논문보다 약함 (완결성 비평).** additive-only(β만, γ 없음) FiLM이고, 주입 위치도 **GDFN(FFN)의 hidden 한 곳뿐** — attention(MDTA)엔 조건화 안 감. 게다가 `drop_path`를 타서 **학습 때 decoder/refinement에선 조건화 FFN 가지가 ~10% 확률로 통째 드롭**(eval엔 항상 on) = 미세한 train/eval 비대칭. 이것도 공식과 동일하지만, "scale+shift adaLN"을 기대하면 이건 더 약한 realization이다.

**요컨대:** UC는 **버그로 망가진 게 아니다.** "논문 그대로의 부품"을, "논문이 노리던 다양성이 없는 문제"에, "참조가 이 경로에선 안 쓰던 방식으로", "10ep 예산에 안 맞는 zero-init 없이" 붙인 것 — **기대이익 ≤ 0, 확정비용 > 0(시간)** 인 레버다.

---

## §1 UC가 코드에서 실제로 도는 경로 (추적, 확정)

`--uc` 켜면 이렇게 흐른다:

1. `train.py:44` `--uc` → `train_part.py:192` `tiny_8gb_args(..., uc=True)` → `config.py:56-58` `time_cond=label_cond=True`.
2. `adapter.py:34` `self.uc=True` → `adapter.py:53-57` forward에서 `self.cascaded(..., timestep=0)` 호출 후 3-튜플이라 `out=out[0]`.
3. `latent_recon.py:109` `actual_timestep = 0*len(recon_models) + i = i` (cascade 번호 0~3).
4. `restormer_mri.forward` `restormer.py:1276-1288`에서 `mask_type·acc·acq`로 label 계산 → `label = acc_idx ∈ {0,1}`. 이걸 `recon_model(x, cas_skips, timestep=i, label=acc_idx)`로 전달.
5. `Restormer.forward` `restormer.py:768-780`: `emb = time_embed(sinusoid(i·50)) + label_emb(sinusoid(acc_idx·10))`.
6. `GDFN.forward` `restormer.py:313-320`: 매 FFN 블록에서 `emb_out = emb_layers(emb)` 만들어 `x = x + emb_out` (additive bias).

grad checkpointing과의 상호작용은 **깨끗함**(비평 확인): `actual_timestep`·label은 파이썬 int로 positional 전달되고 emb는 checkpoint 안에서 재계산 → dead-grad·이중적용 없음. `test_sdum_uc.py::test_uc_conditioning_is_actually_used`가 UC 파라미터에 grad가 실제로 흐르는지 검사해 "조용한 무동작"은 막혀 있다. **즉 "UC가 안 돈다"는 아니다 — 돌긴 잘 돈다.**

---

## §2 "논문대로인가?" — 항목별 판정표 (전부 코드 확정)

| # | 항목 | 판정 | 근거(파일:줄) | 성격 |
|---|---|---|---|---|
| F1 | UC 부품 코드 | ✅ 참조와 동일 | `restormer.py` 30-50·302-320·741-780·1287 = 공식과 import 2줄 빼고 byte-identical | 정보 |
| F2 | timestep=0로 index 경로 활성 | ⚠️ 참조와 divergence | 공식 `train.py:348,570`은 timestep 없이 호출 → `latent_recon.py:109` actual_timestep=None으로 죽음. repo `adapter.py:56`은 timestep=0으로 깨움. 참조는 flow에서만 index 씀(`latent_recon.py:235,248`) | 정합성 이탈 |
| F3 | label 1비트 붕괴 | ⚠️ 구조적 부적합 | `config.py:45-47`(mask 1·acq 1·acc 2) → `restormer.py:1287` label=acc_idx∈{0,1} | 구조 |
| F4 | time↔label 강제결합 | ⚠️ latent trap | `restormer.py:659`(emb_channels=None if not time_cond)·`748-754`(label_emb가 그 값 사용) → label-only면 `Linear(C,None)` 크래시. `config.py:56-58`이 둘 다 켬 | 함정 |
| F5 | cascade-index 중복 | ✅ 중복 맞음 | `latent_recon.py:284,308-316`(cascade별 독립 인스턴스)·`config.py:42`(flow=False)·`adapter.py:56`(timestep=0→index 상수) | 정보 |
| F6 | 조건화 zero-init 없음 | ⚠️ 구조적 부적합 | `restormer.py:303-309·742-746·750-754` 전부 기본 init, `320` x+=emb_out. (공식도 동일) | 구조 |
| F9 | label stride 인덱싱 | 🐛 latent(현재 무해) | `restormer.py:1287` mask stride가 `len(mask_types)`(=n_mask*n_acq), 정답은 `len(acc_factors)*len(acq_types)`(=n_acc*n_acq). mask_idx=0이라 안 터짐. 공식도 동일 | 함정 |
| +1 | recon 경로 `--uc` 수동 | ⚠️ medium footgun | `test_part.py:87-89`(CLI args로 빌드) vs `test_part.py:22-28`(체크포인트 saved.uc로 빌드). 전자만 자기기술 아님 → 플래그 빠뜨리면 strict load 크래시 | 함정 |
| +2 | num_classes 死파라미터 | ℹ️ 오해유발 | `restormer.py:1048-1052`가 num_classes=2 계산하지만, 활성 label_emb는 sinusoid→MLP(`750-754`)라 num_classes를 **안 씀**(쓰는 건 주석처리된 `LabelEmbedder`, `749`). label은 2-class 범주가 아니라 **연속 스칼라 acc_idx·10** | 정보 |
| +3 | additive-only FiLM + drop_path | ℹ️ 논문보다 약함 | `restormer.py:313-320`(β만, γ 없음)·FFN에만 주입(attn 제외 `453/431`)·`decoder/refinement`는 drop_path 0.1로 조건화 가지 ~10% 드롭 | 정보 |
| +4 | sinusoid 라벨 임베딩 퇴화 | ℹ️ 저정보 | `restormer.py:777-780`: acc4→y=0→sinusoid=[1..1,0..0] 상수벡터, acc8→y=10만 일부 고주파 채널이 움직임. 실효 조건화 차원이 channels[0](16/32)보다 훨씬 작음 | 정보 |

정직성 메모: F1은 실제로 `diff` 떠서 **차이 2줄(import 경로)** 확인했고, 나머지도 전부 파일:줄로 재검증했다(적대적 반박 시도까지 거침). "의심"으로 남는 건 §3의 acc8 인과뿐이다.

---

## §3 그래서 run12에서 왜 퇴행했나 — 특히 acc8 (여기부터 일부 의심)

먼저 **정직하게**: run12는 `--fg-loss --bbox-loss-lambda 0.5 --uc` **3레버 묶음**(`shell3.sh:77-81`)이라, 3축 퇴행(full −0.0047·bbox −0.0050·시간 +21ms)이 **bbox 탓인지 UC 탓인지 코드만으론 분리 불가**다. run12 md 자신도 그래서 run13(run11+UC만)을 판별용으로 돌리는 중이라고 못박았다. 이 문서는 그 confound를 **UC 쪽만** 코드로 뜯는다.

**코드로 확정할 수 있는 UC의 순손해 (의심 아님):**

- **시간 +21ms는 UC 때문이 거의 확실.** UC가 얹은 param = cascade마다 `time_embed`+`label_emb` MLP + **모든 GDFN 블록의 emb_layers**(F4·F5·비평 +3). bbox 손실은 param을 안 늘리니(손실항일 뿐), **1514→1535ms의 +21ms는 UC param 몫**으로 보는 게 자연스럽다. 채점은 시간축이 1급 지표라, **UC는 SSIM이 본전이어도 시간에서 확정으로 진다.**
- **SSIM 기대이익 ≈ 0.** UC가 주는 신호는 acc4/acc8 1비트(F3)뿐이고, 그 1비트조차 zero-filled 입력·DC에 이미 암묵적으로 들어있다(모델이 언더샘플 정도를 입력에서 이미 봄). 새 정보가 사실상 없다.

**acc8이 더 타격받은 이유 (전부 "의심"·코드기반 가설 — 확정 아님):**

run12 md 기준 acc8 하락(full −0.0067·bbox −0.0059)이 acc4(−0.0029·−0.0041)보다 크다. 이게 UC 때문이라면 코드가 시사하는 메커니즘은 셋이다. 셋 다 **가설**이고, acc8 숫자 자체가 bbox와 confound돼 있으니 **run13이 나와야 확정**된다.

1. **용량 쪼개기 (capacity split).** UC가 acc를 명시 label로 주면, 작은 모델([16,32]×4·10ep)이 acc4-모드/acc8-모드로 **행동을 분할**하도록 유도된다. 그런데 acc8은 더 어려워서 **더 많은** 용량·데이터가 필요한데, 분할은 각 모드의 실효 표본·용량을 **줄인다** → 어려운 쪽(acc8)이 더 손해. (근거는 구조: label이 acc를 직접 가르니 분할 유인이 생김. "실제로 분할했다"는 미측정 = 의심.)
2. **zero-init 없는 워밍업 비용 (F6).** UC-on은 0스텝부터 GDFN마다 랜덤 bias를 주입해 **무조건화 baseline에서 이탈한 지점**에서 출발한다. 10ep밖에 안 도니 이 perturbation을 다 못 씻어낸다. 여유(margin)가 적은 어려운 acc8이 회복을 덜 한다. (perturbation 크기는 기본 init 스케일 ~1/√fanin이라 **가벼운** 비용 = 이게 주범이라 보긴 약함, 보조 요인.)
3. **저정보·비대칭 조건화 (비평 +4).** label 신호가 acc4=상수벡터/acc8=일부 채널만 움직임이라, 결국 "acc8 쪽에만 약한 bias 하나 더" 얹는 꼴에 가깝다. 이 약한 bias가 도움이 안 되면 어려운 acc8에서 손해로 나타나기 쉽다.

**균형 잡힌 해석:** SSIM 퇴행(acc8 포함)의 **주범은 bbox λ=0.5**일 가능성이 더 크다 — run11에서 fg-loss가 이미 bbox를 올려둔 상태라 bbox 가중손실의 한계효용이 음수라는 게 run12 md의 결론이고, 코드상 UC의 SSIM 영향은 "작은 perturbation" 수준이다. **UC의 확정 죄목은 "시간 +21ms"이고, acc8 SSIM 추가하락에 UC가 얼마나 기여했는지는 run13로만 갈린다.**

---

## §4 코드 관점 권고

- **P0 — run13(run11+UC만) 채점으로 UC 순효과 확정.** run13 ≈ run11이면 UC는 (시간 빼고) SSIM 중립 → run12 퇴행의 SSIM 원흉은 bbox. run13 < run11이면 UC도 SSIM에 유해. 어느 쪽이든 **시간 +21ms 손해는 남는다.**
- **P0 — UC 폐기가 유력, 그리고 그게 원래 SPEC과 일치.** `confirmed_model.md`/config 주석의 확정 설계가 애초에 "UC 제거(time_cond=label_cond=False)"였다(`config.py:35-37`). knee는 조건 신호가 1비트라 논문 이득이 안 옮겨오고, 코드상 순비용(시간)만 확실하다. "빼기"가 이 지점의 정답에 가깝다.
- **P2 — 그래도 UC를 살릴 거면(ROI 낮음, 참고용):**
  - `label`만 쓰고 `timestep`(cascade-index)은 끄고 싶은데 **지금 코드론 불가**(F4 결합). 살리려면 `restormer.py:659`의 emb_channels 할당을 `time_cond or label_cond` 기준으로 풀어 label-only를 빌드 가능하게 고쳐야 함.
  - 조건화 마지막 Linear를 **zero-init**(F6)하면 10ep 예산에서 baseline 이탈 비용이 사라져 최소한 "손해는 안 보는" 출발이 된다.
  - `reconstruct.py` 추론 경로를 `load_model`처럼 **체크포인트 saved.uc로 자기기술화**(비평 +1)해 플래그 footgun 제거.
  - 단 위 셋을 다 해도 **신호가 1비트라는 F3는 안 바뀐다** → 기대이익 상한이 낮다.

---

## §5 확정 vs 의심 (한눈)

- **확정(코드):** UC 부품=참조 동일(F1) · timestep=0 배선은 참조와 다름(F2) · label 1비트(F3) · time↔label 결합(F4) · index 중복(F5) · zero-init 없음(F6) · stride latent 버그(F9) · recon `--uc` footgun · num_classes 死파라미터 · additive-only+drop_path · sinusoid 퇴화. **시간 +21ms = UC param 몫**(강한 정황).
- **의심(가설·미측정):** run12 acc8 추가하락에서 **UC의 기여분**. 코드는 "왜 UC가 도움 안 되고 어려운 쪽을 더 깎을 수 있는지"의 메커니즘만 준다. **정량 분리는 run13 채점이 나와야 한다.**
