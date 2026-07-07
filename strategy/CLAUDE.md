# strategy/ — Claude 작업 규칙

이 폴더에서 전략 문서를 읽거나 고칠 때 지킬 것. (사람용 개요는 [`README.md`](README.md), 대회 배경 전체는 루트 [`../CLAUDE.md`](../CLAUDE.md).)

## 문서 지도 (single source of truth)

각 문서는 **담당 범위가 고정**돼 있다. 새 내용은 반드시 담당 문서에만 쓴다.

| 문서 | **담는 것** | **담지 않는 것** |
|---|---|---|
| [`pipeline.md`](pipeline.md) | 확정 재구성 파이프라인 = crop·정규화·증강·감도맵·GRAPPA prior·출력·채점. **백본 무관 고정 계약. 여기가 SPEC 단일 기준.** 실측 근거표 포함. | 백본 종류·백본별 세부(→backbone), 미확정 아이디어(→decisions) |
| [`backbone.md`](backbone.md) | 현재 백본(SDUM) 결정 노브 + **교체 이력 표** + PromptMR+ 축소 선례. 모델 바꾸면 여기만 수정. | 파이프라인 고정 계약(→pipeline) |
| [`implementation.md`](implementation.md) | 코드 배선 상태(✅구현/⏳보류/⬜미구현)·의존성 순서·소비부 계약. | 결정 근거(→research), 왜 그 값인지 |
| [`decisions.md`](decisions.md) | **Part 1** 열린 후보(미확정 A/B) / **Part 2** 제외 항목 + 이유. | 확정 사항(→pipeline/backbone) |
| [`open_issues.md`](open_issues.md) | 코드↔문서 갭 감사. **Part A = 코드로 확인된 사실 / Part B = 미검증 가설**. | — |
| [`research/`](research/) | 근거·배경·논문: model_strategy·augmentation·preprocessing·related_work. | 확정 스펙 재정의(참조만) |

## 🆕 신규 정보 라우팅 (어디에 · 어떻게)

새 정보가 들어오면 **성격을 판별 → 담당 문서 한 곳에만** 기록한다.

| 새 정보 | 적을 곳 | 작성 방법 |
|---|---|---|
| 확정된 파이프라인 사실(모델 무관) | `pipeline.md` | 해당 섹션 갱신. 실측이면 근거표에 **수치+날짜**. |
| 백본 교체 / 백본별 결정 | `backbone.md` | "현재 백본" 갱신 + **교체 이력 표에 행 추가**. pipeline.md §5는 "backbone.md 참조"로 **그대로 둔다**(모델명 하드코딩 금지). |
| 해볼 만한 아이디어(미확정) | `decisions.md` Part 1 | 항목·판정 게이트·근거를 표행으로. **SSIM_bbox 비퇴행이 기본 게이트**, incremental A/B 원칙. |
| 안 쓰기로 결정 | `decisions.md` Part 2 | **반드시 제외 이유 명시**(SSIM_bbox 파괴/hallucination/물리무효/데이터제약 중 무엇). |
| 코드 배선 상태 변화 | `implementation.md` | ✅/⏳/⬜ 및 소비부 튜플 계약 갱신. |
| 코드↔문서 불일치·놓친 점 | `open_issues.md` | **Part A(확인된 사실)** vs **Part B(가설)** 구분해 기재. |
| 결정의 근거·배경·논문 | `research/`의 해당 파일 | 근거·출처만. 여기서 스펙을 확정하지 말 것(확정은 pipeline/backbone). |

**생애주기(상태 전이) 규칙**
- **후보 → 확정**: `decisions.md`에서 제거하고 `pipeline.md`(또는 backbone.md)에 정식 기재. 양쪽에 중복 방치 금지.
- **확정 → 번복**: 번복 사유를 남기고 관련 문서 동기화. 백본 번복이면 backbone.md 이력 표에 기록.
- **가설(open_issues Part B) → 검증됨**: 사실이면 Part A 또는 담당 문서로 승격, 아니면 decisions.md 제외로.

## 작성 원칙 (항상)
1. **한 곳에서만 정의, 나머지는 링크.** 같은 사실을 복붙하면 한 곳만 낡는다.
2. **백본을 pipeline.md에 하드코딩 금지.** 백본은 교체 가능 → backbone.md에서만. pipeline §5는 참조로 유지.
3. **검증된 판단을 재검증하지 말 것.** decisions.md(제외 이유)·pipeline.md(실측표)·open_issues.md Part A는 이미 확인·결정된 것 → 근거로 바로 활용, 재검증에 시간·토큰 쓰지 말 것.
4. **텍스트를 함부로 지우지 말 것.** 검증 끝난 기록은 보존. 정리는 병합·이동으로, 삭제는 진짜 중복(리다이렉트 껍데기)만.
5. **날짜·이유를 남길 것.** 특히 확정/제외/번복.

## 하드 제약 (제안·구현 시 항상 전제)
- 무릎(knee) 전용 · 채점 = SSIM_full + **SSIM_bbox(순위 결정축)** + 추론시간.
- **GTX 1080 8GB · FP32**(Tensor Core 없음 → AMP 이득 없음). A100급 세팅 그대로 못 씀.
- **외부 데이터·사전학습 가중치 금지**(구조만 사용, 처음부터 학습).
- 최종 제출은 주최측 서버에서 **고정 seed 재현** 검증 → 재현성 필수.
- **SSIM_bbox를 해치는(병변 hallucination) 변경은 화질이 올라도 채택 금지.**

## 구조 유지 & Git 규칙
- **기본은 현 구조 유지.** 사용자의 별도 지시가 없으면 `strategy/`의 현재 파일 구조(pipeline·backbone·implementation·decisions·open_issues·research/ + 이 문서·README)를 **그대로 사용**한다. 파일을 새로 만들거나 옮기거나 이름을 바꾸거나 분할·병합하지 말 것 — 내용만 담당 문서에 추가한다.
- **Git 병합 시에도 동일.** GitHub에서 merge·rebase·pull로 남의 변경과 합칠 때도 이 구조를 보존한다. **충돌이 나도 내용만 합치고, 파일 구성·이름·역할 분리는 바꾸지 않는다**(구조 변경은 사용자 지시가 있을 때만).
- **push/commit 명령을 받으면 알릴 것.** 사용자가 GitHub push·commit을 지시하면, 실행 전에 **"현재 strategy 구조를 그대로 유지한 채 진행한다"**는 방침을 한 줄로 알린 뒤 진행한다.

> **알림 예시** (push/commit 지시를 받았을 때):
> "커밋·푸시 진행합니다. strategy 폴더는 현 구조(pipeline·backbone·decisions·open_issues·research/)를 **그대로 유지**하며, 병합 충돌이 있어도 내용만 합치고 구조는 바꾸지 않습니다."

## 링크 규칙
- 상대 링크 유지. `research/` 안 문서에서 상위 문서는 `../pipeline.md`처럼.
- 파일을 옮기거나 이름을 바꾸면 참조 링크를 함께 갱신하고, 끝나면 `grep`으로 **구 파일명 잔여 0 + 링크 대상 실재**를 검증할 것.
