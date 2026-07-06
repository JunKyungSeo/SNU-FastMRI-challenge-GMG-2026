# CLAUDE.md — 이 repo에서 작업하는 Claude에게

이 파일은 **미래 세션(GPU 서버 포함)의 Claude가 먼저 읽어야 하는 핸드오프 노트**다. 특히 SDUM 백본 이식 코드는 **개발 환경에서 작성만 됐고 한 번도 실행되지 않았다.** 아래 순서를 반드시 지킬 것.

---

## 0. 현재 환경 구분 (중요)

- **지금까지의 작업 환경 = 개발용 로컬(Windows 노트북)**: torch/MONAI **미설치**, GPU 없음(MX450 2GB뿐), 데이터 없음.
  - 그래서 **로컬에서 한 검증 = 파이썬 AST 문법 파싱뿐.** `pytest`는 **한 번도 돌리지 않았다.**
- **GPU 서버로 이동하면 사용자가 명시적으로 알려준다.** 그 신호가 오기 전까지는 여기가 개발 환경이라고 가정.

---

## 1. GPU 서버로 옮기면 — ⚠️ 학습보다 먼저 "모델 돌려서 검증"

사용자가 "이제 GPU 서버다"라고 하면, **본 학습(train) 전에 아래 검증을 먼저** 수행한다. 순서대로:

```bash
# (0) 가상환경 권장 (전역 오염 방지). Python 3.12.x 권장(repo 타깃 3.12.9).
pip install -r requirements.txt          # torch/monai/einops/timm/pytest 등

# (1) import & 빌드 스모크
pytest tests/test_sdum_build.py -v       # 의존성 import + 모델 빌드 + 파라미터 수

# (2) ⭐ 어댑터 게이트 (가장 중요) — 이게 통과해야 그 다음으로 감
pytest tests/test_sdum_adapter.py -v     # forward shape (B,384,384) + finite + 1-step 학습

# (3) 8GB 적재 스윕 (실제 GPU에서만 의미)
python scripts/probe_oom.py --cascades 6 --channels 24 48
#   -> peak GB 또는 OOM 출력. peak ~7.5GB 이하로 OK인 가장 큰 조합 선택.

# (4) 위가 다 통과한 뒤에야 실제 학습
python train.py --model sdum --num-cascades 4 --channels 32 64 \
    -t <train_dir> -v <val_dir> -n sdum_test ...
```

**검증(1)~(3)이 통과하기 전에는 장시간 학습을 돌리지 말 것.**

---

## 2. 솔직한 리스크 (미검증 지점)

개발 환경에서 실행을 못 했기 때문에 아래는 **런타임 미검증**이다. GPU에서 처음 돌릴 때 오류가 날 수 있다:

- **가장 위험: 어댑터(`utils/model/sdum/adapter.py`).** 우리 파이프라인은 k-space `(B,C,H,W,2)` + mask `(B,1,1,W,1)`를 주는데, NV SDUM 모델은 **영상 도메인 `(B,T,C,H,W,2)`**를 받는다. 어댑터가 ifft·T=1 차원·mask broadcast·RSS·384 crop으로 잇는다.
  - **불확실 포인트 1**: NV `restormer_mri.forward`의 mask가 `"b t c h w two -> (b t) c h w two"`로 rearrange되는데, 우리 `mask_b`(B,1,C,H,W,2) broadcast가 `soft_dc`/`get_acs_region`의 기대 shape과 정확히 맞는지.
  - **불확실 포인트 2**: cascaded 출력이 영상 도메인 multicoil `(B,T,C,H,W,2)`이 맞는지(그래서 RSS로 합쳐 `(B,384,384)`가 되는지). 만약 coil-combined 단일 채널을 반환하면 RSS를 단순화해야 함.
  - **증상**: `test_sdum_adapter`가 shape/rearrange 에러로 실패. → **수정 가이드는 플랜의 Task 3 ⚠️ 노트** 참조: `docs/superpowers/plans/2026-07-06-sdum-backbone-port.md`.
- **의존성 해소 미검증**: NV 코드는 MONAI에 깊게 의존. `nv_utils.py`는 필요한 헬퍼 6개만 AST로 추출했는데, 실제 import 시 빠진 이름이 있으면 `ImportError`가 날 수 있음(그땐 `NV-SDUM-official/scripts/utils.py`에서 해당 함수를 추가 vendor).
- **MONAI 버전 호환**: `monai.apps.reconstruction.*` 경로가 설치된 MONAI 버전에서 동일한지 확인(≥1.3 가정).

---

## 3. 이 이식에서 확정된 설계 (배경)

- 기존 **E2E-VarNet 백본을 SDUM으로 교체**. `--model sdum`으로 선택(기본은 여전히 `varnet`이라 기존 경로 안 깨짐).
- 제약: **GTX 1080 8GB FP32**(Muon/BF16 안 씀 → **AdamW FP32**), knee 전용, acc4/acc8, 출력 384², 순위축 SSIM_bbox.
- **사전학습 가중치/외부데이터 금지**(대회 규정) → NV HuggingFace 자동다운로드 코드는 vendor 안 함. 처음부터 학습.
- 확정 결정(상세: `strategy/model_decision_points.md`): gradient checkpointing ON(NV cascade 루프 내장) · CSME per-cascade · SWDC 유지 · **UC 제거**(`time_cond/label_cond=False`) · batch=1 · coil compression 보류 · num_frames=1(2D).
- **레버 스윕 대상**: `--num-cascades`(T), `--channels`(Restormer 폭). config 기본 `[32,64]`=2-level; `[C]` 하나면 1-level A/B(병변 과평활 대비, 상세: model_decision_points.md).

## 4. 범위 밖 (별도 작업/다른 팀원)

- **GRAPPA prior + residual** (다른 팀원 진행 중), **bbox 가중 손실**, SWDC 패턴 튜닝, CSME 옵션(c) 주기적 재추정, progressive expansion. 상세: `strategy/` 문서들 + 플랜의 "범위 밖" 절.

## 5. 참고 문서

- 이식 플랜(태스크·수정 가이드): `docs/superpowers/plans/2026-07-06-sdum-backbone-port.md`
- 모델 결정 로그: `strategy/model_decision_points.md`
- 원본 SDUM 코드(참고용, repo 밖): `../NV-SDUM-official/` (NVIDIA-Medtech/NV-Raw2insights-MRI, Apache-2.0)
- SDUM 논문: arXiv:2512.17137
