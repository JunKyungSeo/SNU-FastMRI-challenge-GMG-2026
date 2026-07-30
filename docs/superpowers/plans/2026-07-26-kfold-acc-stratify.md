# K-Fold acc 층화 분할 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `kfold_split`을 acc 층화 분할로 교체해, `--acc all/4/8` 어느 값으로 호출해도 같은 볼륨이 항상 같은 fold에 배정되게 한다 (MoE pretrain→미세조정 파이프라인의 hold-out 오염 제거).

**Architecture:** acc 그룹(acc4/acc8)마다 각자의 `RandomState(seed)`로 독립 셔플 후 각각 k등분, fold i = 그룹별 i번 조각의 합집합. `--acc 4/8`은 조립된 fold에서 해당 그룹만 남긴다(경계 불변). 순수 헬퍼 `_stratified_folds`로 분할 로직을 분리해 파일시스템 없이 테스트한다. 기본 동작 교체(플래그 없음) — 기존 비층화 파티션(promptmr_kfold_f1/f2)과는 hold-out이 달라지며, 이는 감수하기로 결정됨. resume/merge 가드·파티션 지문 로그는 **범위 밖**(설계 리뷰에서 제외 결정).

**Tech Stack:** Python + numpy (`RandomState`, `array_split`), pytest (서버 실행), bash 큐 슬롯 주석.

## Global Constraints

- **브랜치**: `jkMRplus` (현재 브랜치)에 직접 커밋. push 하지 않는다.
- **커밋 전 반드시 사용자에게 "지금 사용자가 누구인가"를 물어 `git config user.{name,email}`(local)을 맞춘 뒤 커밋** (repo CLAUDE.md §9 규칙).
- **커밋 메시지·어디에도 Claude/AI attribution 금지** (`Co-Authored-By: Claude`, `🤖 Generated with…` 절대 불가 — repo CLAUDE.md §9가 기본 지시보다 우선).
- **로컬에 torch 없음** → pytest는 이 머신에서 실행 불가. 로컬 게이트 = `python -m py_compile <파일>`. pytest·스모크는 Task 4에서 서버(사용자) 몫. (기존 관례: 커밋 ecbf6b7 "서버 검증 게이트")
- 주석·docstring은 한국어, 각 파일의 기존 문체를 따른다. worklog.md는 이모지·별표(강조) 금지, 완결 문장, 최신 엔트리가 구분선 바로 아래.
- `kfold_split`의 반환 인터페이스 `(train_ids, val_ids)`(str set 2개)는 불변 — 소비부(`create_kfold_data_loaders`) 무수정.

---

### Task 1: `_stratified_folds` 순수 헬퍼 + 단위 테스트

**Files:**
- Modify: `utils/data/load_data.py` (kfold_split 바로 위, 현재 275행 `kfold_roots` 앞에 헬퍼 추가)
- Create: `scripts/tests/test_kfold_stratify.py`

**Interfaces:**
- Consumes: 없음 (numpy만 사용, 기존 코드와 독립)
- Produces: `_stratified_folds(ids_by_group: dict[str, list[str]], k: int, seed: int) -> list[set[str]]` — 길이 k의 fold 리스트. Task 2의 `kfold_split`이 호출.

- [ ] **Step 1: 테스트 파일 작성 (단위 테스트 4개)**

`scripts/tests/test_kfold_stratify.py` 생성 (conftest.py가 repo 루트를 sys.path에 넣어주므로 별도 경로 조작 불필요):

```python
"""kfold_split acc 층화 분할 테스트 (2026-07-26 설계).

핵심 불변식: --acc all/4/8 어느 값으로 분할해도 같은 볼륨은 항상 같은 fold.
kfold_split은 (root/"kspace").glob("*.h5")의 **파일명만** 보고 내용은 안 열므로
tmp_path에 빈 파일 touch로 실데이터 없이 검증한다.

⚠️ 로컬에 torch가 없으면 import에서 죽는다 — 서버에서 실행:
  python -m pytest scripts/tests/test_kfold_stratify.py -v
"""
from types import SimpleNamespace

import numpy as np
import pytest

from utils.data.load_data import _stratified_folds, kfold_split


# ---------- _stratified_folds 단위 테스트 (파일시스템 불필요) ----------

def _ids(acc, n, root="train"):
    return [f"{root}__knee_acc{acc}_{i}.h5" for i in range(n)]


def test_group_assignment_independent_of_other_groups():
    """한 그룹의 fold 배정은 다른 그룹의 존재와 무관 — 층화의 존재 이유."""
    both = _stratified_folds({"4": _ids(4, 30), "8": _ids(8, 30)}, k=5, seed=430)
    only4 = _stratified_folds({"4": _ids(4, 30)}, k=5, seed=430)
    for i in range(5):
        assert {v for v in both[i] if "acc4" in v} == only4[i]


def test_folds_balanced_per_group():
    """100+100을 k=5로 → fold마다 정확히 acc4 20 + acc8 20."""
    folds = _stratified_folds({"4": _ids(4, 100), "8": _ids(8, 100)}, k=5, seed=430)
    for f in folds:
        assert sum(1 for v in f if "acc4" in v) == 20
        assert sum(1 for v in f if "acc8" in v) == 20


def test_deterministic_same_seed():
    a = _stratified_folds({"4": _ids(4, 40), "8": _ids(8, 40)}, k=5, seed=430)
    b = _stratified_folds({"4": _ids(4, 40), "8": _ids(8, 40)}, k=5, seed=430)
    assert a == b


def test_group_smaller_than_k_raises():
    with pytest.raises(ValueError, match="k="):
        _stratified_folds({"4": _ids(4, 3)}, k=5, seed=430)
```

- [ ] **Step 2: 헬퍼 구현**

`utils/data/load_data.py`의 `kfold_roots`(현재 275행) 바로 위에 추가:

```python
def _stratified_folds(ids_by_group, k, seed):
    """acc 그룹별 독립 셔플 → 각각 k등분 → fold별 합집합. 길이 k의 set 리스트 반환.

    각 그룹이 **자기만의** RandomState(seed)로 섞이므로 한 그룹의 fold 배정은
    다른 그룹의 존재 여부와 무관하다 — --acc 필터가 fold 경계를 못 움직이는 근거.
    그룹 순회는 sorted(키) 순서라 결정적이다.
    """
    folds = [set() for _ in range(k)]
    for key in sorted(ids_by_group):
        ids = np.array(sorted(ids_by_group[key]))
        if len(ids) < k:
            raise ValueError(
                f"K-Fold 층화: acc{key} 볼륨 {len(ids)}개 < k={k} — 빈 fold가 생긴다.")
        shuffled = ids[np.random.RandomState(seed).permutation(len(ids))]
        for i, chunk in enumerate(np.array_split(shuffled, k)):
            folds[i].update(map(str, chunk))
    return folds
```

- [ ] **Step 3: 로컬 문법 게이트**

Run: `python -m py_compile utils/data/load_data.py scripts/tests/test_kfold_stratify.py`
Expected: 출력 없이 exit 0. (pytest는 로컬 torch 부재로 Task 4에서 서버 실행.)

- [ ] **Step 4: 커밋**

사용자에게 현재 작업자가 누구인지 확인하고 `git config user.name/user.email`(local) 설정 후:

```bash
git add utils/data/load_data.py scripts/tests/test_kfold_stratify.py
git commit -m "feat(kfold): acc 층화 분할 헬퍼 _stratified_folds + 단위 테스트"
```

---

### Task 2: `kfold_split` 재배선 + 통합 테스트

**Files:**
- Modify: `utils/data/load_data.py:280-317` (`kfold_split` 본체·docstring)
- Modify: `scripts/tests/test_kfold_stratify.py` (통합 테스트 추가)

**Interfaces:**
- Consumes: Task 1의 `_stratified_folds(ids_by_group, k, seed) -> list[set[str]]`, 기존 `_file_acc(name) -> str|None`, `_norm_acc(acc) -> str|None`, `volume_id(root, name) -> str`, `kfold_roots(args) -> list[Path]`
- Produces: `kfold_split(args) -> (train_ids: set[str], val_ids: set[str])` — 시그니처·반환형 기존과 동일 (소비부 무수정)

- [ ] **Step 1: 통합 테스트 추가**

`scripts/tests/test_kfold_stratify.py` 끝에 추가:

```python
# ---------- kfold_split 통합 테스트 (tmp_path에 빈 h5 touch) ----------

def _make_data(tmp_path, n4=50, n8=50):
    """train/·val/ 두 root에 같은 이름의 acc4/acc8 파일을 touch.
    실데이터처럼 두 root에 동명 볼륨이 존재 → volume_id가 구분해 총 2*(n4+n8)개."""
    for root in ("train", "val"):
        d = tmp_path / root / "kspace"
        d.mkdir(parents=True)
        for i in range(n4):
            (d / f"knee_acc4_{i}.h5").touch()
        for i in range(n8):
            (d / f"knee_acc8_{i}.h5").touch()
    return tmp_path


def _args(tmp_path, acc="all", k=5, fold=0, seed=430):
    return SimpleNamespace(
        data_path_train=tmp_path / "train", data_path_val=tmp_path / "val",
        k_fold=k, val_fold=fold, kfold_seed=seed, acc=acc)


def test_acc_filter_keeps_fold_boundaries(tmp_path):
    """핵심 불변식: --acc 4/8의 분할 == --acc all의 분할에서 그 acc만 남긴 것."""
    _make_data(tmp_path)
    for fold in range(5):
        tr_all, va_all = kfold_split(_args(tmp_path, "all", fold=fold))
        tr_4, va_4 = kfold_split(_args(tmp_path, "4", fold=fold))
        tr_8, va_8 = kfold_split(_args(tmp_path, "8", fold=fold))
        assert va_4 == {v for v in va_all if "acc4" in v}
        assert tr_4 == {v for v in tr_all if "acc4" in v}
        assert va_8 == {v for v in va_all if "acc8" in v}
        assert tr_8 == {v for v in tr_all if "acc8" in v}


def test_partition_complete_disjoint_balanced(tmp_path):
    """5개 val fold가 전체 200 볼륨을 정확히 1회씩 커버 + train/val 서로소 + fold마다 acc 20/20."""
    _make_data(tmp_path)   # 총 200 = acc4 100 + acc8 100
    seen = set()
    for fold in range(5):
        tr, va = kfold_split(_args(tmp_path, fold=fold))
        assert tr.isdisjoint(va)
        assert len(tr) + len(va) == 200
        assert len(va) == 40
        assert sum(1 for v in va if "acc4" in v) == 20
        assert sum(1 for v in va if "acc8" in v) == 20
        seen |= va
    assert len(seen) == 200


def test_kfold_split_deterministic(tmp_path):
    _make_data(tmp_path)
    assert kfold_split(_args(tmp_path)) == kfold_split(_args(tmp_path))


def test_file_without_acc_token_raises(tmp_path):
    """acc 토큰 없는 h5가 섞이면 fail-fast (이 데이터셋은 acc4/acc8 전수 규약)."""
    _make_data(tmp_path)
    (tmp_path / "train" / "kspace" / "weird_noacc.h5").touch()
    with pytest.raises(ValueError, match="accN"):
        kfold_split(_args(tmp_path))


def test_acc_filter_no_match_raises(tmp_path):
    _make_data(tmp_path, n8=0)   # acc8 파일 없음
    with pytest.raises(ValueError):
        kfold_split(_args(tmp_path, acc="8"))
```

- [ ] **Step 2: `kfold_split` 재작성**

`utils/data/load_data.py`의 `kfold_split` 전체(현재 280~317행)를 다음으로 교체. docstring의 "⚠️ --acc 오염 경고" 문단(287~291행)은 불변식 설명으로 대체된다:

```python
def kfold_split(args):
    """K-Fold 분할 → (train_ids, val_ids). 원소는 volume_id(고유 볼륨 식별자).

    **분할 단위는 volume**이다. train/과 val/에 이름이 같은 볼륨이 30쌍 있지만 서로 남남이라
    (예: train/…/knee_acc4_1.h5 = 23슬라이스, val/…/knee_acc4_1.h5 = 31슬라이스) 각각 독립
    단위로 다룬다 → fold마다 정확히 200/k개가 떨어진다.

    **acc 층화(2026-07-26~)**: acc4·acc8 볼륨을 _stratified_folds로 **각각** k등분해 fold별로
    합친다. 그룹별 셔플이 서로 독립이라 --acc all/4/8 중 뭘 줘도 **같은 볼륨은 항상 같은
    fold** → MoE의 pretrain(--acc all)→미세조정(--acc 4/8)을 같은 hold-out으로 이어도
    채점셋이 오염되지 않는다. (구버전은 "필터 후 분할"이라 경계가 --acc에 종속 —
    실측 40개 중 30개 오염. 그 코드는 이 층화로 대체됨.)
    ⚠️ 구버전(비층화)으로 나눈 기존 런(promptmr_kfold_f1/f2)과는 hold-out이 다르다.
    그 런들과 val 숫자를 나란히 비교하지 말 것.

    난수는 seed만의 함수(RandomState(kfold_seed))라 같은 seed면 언제 돌려도 같은 분할이 나온다.
    """
    k = int(getattr(args, "k_fold", 5))
    val_fold = int(getattr(args, "val_fold", 0))
    seed = int(getattr(args, "kfold_seed", 430))
    if not 0 <= val_fold < k:
        raise ValueError(f"--val-fold({val_fold})는 0 이상 --k-fold({k}) 미만이어야 합니다.")
    acc_want = _norm_acc(getattr(args, "acc", "all"))

    roots = kfold_roots(args)
    ids_by_group = {}   # acc 토큰('4'/'8') → volume_id 리스트
    no_token = []
    for r in roots:
        for f in sorted((r / "kspace").glob("*.h5")):
            acc = _file_acc(f.name)
            if acc is None:
                no_token.append(str(f))
                continue
            ids_by_group.setdefault(acc, []).append(volume_id(r, f.name))
    if no_token:
        raise ValueError(
            "K-Fold 층화: 파일명에 accN 토큰이 없어 그룹을 정할 수 없습니다:\n  "
            + "\n  ".join(no_token))
    if not ids_by_group:
        raise ValueError(f"K-Fold: 데이터 경로({roots})에서 h5 파일을 찾지 못했습니다.")

    folds = _stratified_folds(ids_by_group, k, seed)

    if acc_want is not None:
        # --acc 4/8: 조립된 fold에서 그 그룹만 남긴다. 경계 자체는 안 움직인다(층화 불변식).
        keep = set(ids_by_group.get(acc_want, []))
        if not keep:
            raise ValueError(f"K-Fold: acc={acc_want} 볼륨이 없습니다({roots}).")
        folds = [f & keep for f in folds]

    val_ids = folds[val_fold]
    train_ids = set().union(*folds) - val_ids
    return train_ids, val_ids
```

- [ ] **Step 3: 로컬 문법 게이트**

Run: `python -m py_compile utils/data/load_data.py scripts/tests/test_kfold_stratify.py`
Expected: 출력 없이 exit 0.

- [ ] **Step 4: 커밋**

```bash
git add utils/data/load_data.py scripts/tests/test_kfold_stratify.py
git commit -m "feat(kfold): acc 층화 분할로 교체 — --acc와 무관하게 fold 경계 고정"
```

---

### Task 3: 큐 슬롯 주석·worklog 갱신

**Files:**
- Modify: `shell_scripting/shell21.sh` (헤더의 "--acc 추가하지 말 것" 경고 블록)
- Modify: `shell_scripting/shell22.sh` (헤더에 파티션 변경 경고 추가)
- Modify: `worklog.md` (새 엔트리, 구분선 `---` 바로 아래)

**Interfaces:**
- Consumes: Task 2의 동작 변경 사실 (층화가 기본, f1/f2는 구파티션)
- Produces: 없음 (문서만)

- [ ] **Step 1: shell21.sh 경고 블록 교체**

shell21.sh에서 아래 블록을:

```
#   ⚠️ --acc 4/8 을 여기에 추가하지 말 것. --acc 를 주면 그 가속률만 모아 다시 5등분하므로
#      fold 경계가 달라진다(단독 실험이면 정상, 다단계로 섞으면 채점셋 오염).
#      MoE를 k-fold로 검증하려면 층화 분할을 먼저 구현해야 한다 — load_data.kfold_split 주석 참고.
```

다음으로 교체:

```
#   ⚠️ (2026-07-26 갱신) kfold_split이 acc 층화로 교체되어 --acc 4/8 을 줘도 fold 경계가
#      안 바뀐다(MoE 다단계 파이프라인 안전). 단, 이 슬롯으로 이미 완주한 f2 런은
#      **교체 전(비층화) 파티션** 산출물이라, 층화 이후 새로 도는 런과 val 숫자를
#      나란히 비교하면 안 된다 — load_data.kfold_split docstring 참고.
```

- [ ] **Step 2: shell22.sh 헤더에 경고 추가**

shell22.sh 헤더의 `[실험 목적]` 블록 안, `⚠️ 이 런의 완주·과적합 여부는 repo에 기록 없음` 줄 바로 아래에 추가:

```
#       ⚠️ (2026-07-26) kfold_split이 acc 층화로 교체되어 지금 코드로 ON arm을 돌리면
#          OFF arm(교체 전 파티션의 f2)과 hold-out이 달라진다 — 이대로 비교하면 A/B 무효.
#          이 A/B를 살리려면 OFF arm(baseline)을 층화 파티션에서 재학습한 뒤 진행할 것.
```

- [ ] **Step 3: worklog.md 엔트리 추가**

`---` 구분선 바로 아래(현재 2026-07-19 엔트리 위)에 추가. 시각은 작성 시점에 `date -u '+%Y-%m-%d %H:%M'`(UTC)과 KST(+9h)로 채우고, 작성자 태그는 커밋 identity 확인 때 같이 확인한다:

```
## 2026-07-26 HH:MM UTC / 2026-07-26 HH:MM KST — @태그

한 줄 요약: kfold_split을 acc 층화 분할로 교체했다. --acc 값과 무관하게 fold 경계가 고정된다.

무엇: acc4·acc8 볼륨을 각각 k등분해 fold별로 합치는 층화 분할을 kfold_split의 기본 동작으로
넣었다. 순수 헬퍼 _stratified_folds와 pytest(scripts/tests/test_kfold_stratify.py, 서버 실행)를
같이 추가했다. shell21·shell22 헤더의 관련 경고도 현행화했다.

왜: 기존 구현은 --acc로 거른 뒤 분할해서, MoE의 pretrain(--acc all)→미세조정(--acc 4/8)을
이으면 단계마다 hold-out이 달라졌다(실측 40개 중 30개 오염, 4런 분석 문서 참고). 층화가
팀 합의된 전제 조건이었고(shell21 주석·4런 분석 §6), 이번에 기본 동작으로 교체했다.
플래그 방식(기본 off)도 검토했으나 실수로 빼먹으면 오염이 재발하므로 기본 교체를 골랐다.

주의: 교체 전 파티션으로 돈 promptmr_kfold_f1/f2와 새 런의 val은 비교 불가다. shell22의
증강 A/B는 OFF arm(f2)이 구파티션이라 baseline 재학습 없이는 무효 — 다음 run 결정에서 처리.

상태: 커밋 <해시 채우기>. 서버 pytest·스모크 검증은 아직(아래 핸드오프).

핸드오프 → @상대: 서버에서 python -m pytest scripts/tests/test_kfold_stratify.py -v 와
SMOKE=1 FOLD=0 bash shell_scripting/shell21.sh 를 돌려 [K-Fold] 로그의 val 구성이
acc4 20 · acc8 20으로 나오는지 확인 부탁한다.
```

- [ ] **Step 4: 커밋**

```bash
git add shell_scripting/shell21.sh shell_scripting/shell22.sh worklog.md
git commit -m "docs(kfold): 층화 교체에 따른 슬롯 경고·worklog 현행화"
```

---

### Task 4: 서버 검증 게이트 (사람 실행 — 이 세션에서 자동화 불가)

**Files:** 없음 (검증만)

**Interfaces:**
- Consumes: Task 1~3의 커밋 전부
- Produces: 검증 통과 확인 (이후 MoE×k-fold 런의 전제)

- [ ] **Step 1: 서버에서 pytest**

Run (서버, torch 있는 환경): `python -m pytest scripts/tests/test_kfold_stratify.py -v`
Expected: 9개 테스트 전부 PASS.

- [ ] **Step 2: 스모크로 분할 로그 확인**

Run (서버): `SMOKE=1 FOLD=0 bash shell_scripting/shell21.sh`
Expected: `[K-Fold]` 로그에 `val 40 vol (acc4 20 · acc8 20)` — 층화 전(예: f2의 22·18)과 달리 정확히 20/20.

- [ ] **Step 3: 결과를 worklog 핸드오프에 반영**

pytest·스모크 결과를 확인했으면 worklog.md의 "상태" 줄을 갱신(검증 완료 표기)하고 커밋:

```bash
git add worklog.md
git commit -m "docs(worklog): 층화 분할 서버 검증 결과 기록"
```

---

## Self-Review 결과

- **스펙 커버리지**: §1 알고리즘→Task 1·2 / §3 문서·주석→Task 3 / §4 에러 처리→Task 1 Step 2(그룹<k)·Task 2 Step 2(no_token·빈 그룹·val_fold 범위) / §5 테스트 5케이스→Task 1(독립성·균등·결정성·그룹<k)+Task 2(불변식·완전성/서로소·fail-fast·빈 그룹) / §2 가드는 사용자 결정으로 제외(Architecture에 명시). 갭 없음.
- **플레이스홀더**: worklog의 시각(`HH:MM`)·커밋 해시·작성자 태그는 실행 시점 값이라 실행 단계에서 채우도록 지시로 명문화 — 설계 결정 미정 항목 없음.
- **타입 일관성**: `_stratified_folds`의 반환 `list[set[str]]`을 Task 2가 그대로 소비, `kfold_split` 반환 `(set, set)` 기존과 동일 확인.
