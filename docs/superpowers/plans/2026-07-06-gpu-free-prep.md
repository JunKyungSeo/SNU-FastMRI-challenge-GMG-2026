# SDUM GPU-Free 개발 작업 (순수 C) Implementation Plan

> **작업자에게:** 태스크 하나씩 실행. 각 스텝 체크박스(`- [ ]`)로 추적. GPU 없이 개발용 로컬에서 수행.
> **범위 정정(2026-07-06):** 이전 버전은 옵션 A(=CPU에서 pytest로 어댑터 검증)를 섞어 넣었었음. 이 플랜은 사용자가 고른 **옵션 C(설치 없이/가볍게 되는 개발 작업)만** 담는다. 어댑터를 실제로 돌려 검증하는 것(옵션 A)은 여기 없음.

**목표:** GPU 확보 전, 설치 없이(또는 최소 설치로) 할 수 있는 준비 작업 — ①스윕 후보를 좁히는 크기/메모리 추정 ②bbox 손실 스캐폴딩 ③vendored 코드 리뷰 — 을 끝내 GPU가 오면 곧바로 진도를 뺀다.

**아키텍처:** 전부 로컬 개발 작업. C1(분석적 추정)·C3(스캐폴딩 작성)·C4(리뷰)는 **무설치**(파이썬 표준 라이브러리 + AST 파싱만). C2(정밀 param 실측)만 CPU venv가 있을 때 하는 **선택 태스크**.

**기술 스택:** 순수 Python(표준 라이브러리), (선택) CPU venv + torch/MONAI.

## Global Constraints

- **환경:** 개발용 로컬(Windows), GPU 없음. **기본 무설치.** C2만 선택적으로 CPU venv.
- **검증 한계:** 무설치 태스크는 **AST 문법 파싱**까지만 로컬 검증(실행은 GPU/venv 환경으로 미룸).
- **커밋:** 브랜치 `SDUM`. push는 지시 시에만.
- **정확성 주의:** C1의 파라미터/메모리 수치는 **근사(상대 비교용)** — 절대 MB가 아니라 "어느 config가 더 큰가/8GB 위험한가"를 가르는 용도. 정밀값은 C2(venv) 또는 GPU probe에서.
- **범위 밖:** 옵션 A(pytest 어댑터 검증), 실제 8GB 스윕·본 학습(GPU), GRAPPA prior(다른 팀원), bbox 손실의 **데이터 파이프라인 플러밍**(annotation emit)·실데이터 검증.

---

## 파일 구조

- `scripts/estimate_sdum_analytical.py` (생성) — 무설치 근사 파라미터/메모리 랭커.
- `scripts/estimate_sdum.py` (생성, 선택) — venv에서 정밀 param 실측.
- `utils/common/loss_function.py` (수정) — `SSIMBboxLoss` 스캐폴딩 추가.
- `docs/sdum_review_notes.md` (생성) — vendored 코드 리뷰 + watch-list.
- `.gitignore` (필요 시) — C2를 할 경우 `.venv/` 제외.

---

## Task C1: 분석적 파라미터/메모리 추정 (무설치)

> config별 대략적 크기를 **표준 라이브러리만으로** 계산해 GPU 스윕 후보를 좁힌다. torch 불필요. 값은 근사(상대 비교용).

**파일:**
- Create: `scripts/estimate_sdum_analytical.py`

**인터페이스:**
- 소비: 없음(순수 계산).
- 산출: `(cascades × channels)` 그리드의 근사 파라미터 수 + 상대 활성값-메모리 점수를 stdout 표로 출력.

- [ ] **스텝 1: 근사 계산 스크립트 작성**

`scripts/estimate_sdum_analytical.py` 생성. Restormer 블록의 주 파라미터(1×1 conv qkv/project, depthwise conv, GDFN 확장)와 down/up/reduce, CSME(고정 U-Net)를 근사식으로 합산. 값은 근사임을 명시:

```python
"""SDUM 근사 파라미터/메모리 랭커 (torch 불필요, 근사값=상대 비교용)."""


def restormer_level_params(c, blocks, mlp):
    # TransformerBlock 근사: MDTA(qkv 1x1: c*3c, dwconv 3x3: 3c*9, proj: c*c)
    mdta = c * (3 * c) + (3 * c) * 9 + c * c
    # GDFN(project_in 1x1: c*2h, dwconv 3x3: 2h*9, project_out 1x1: h*c), h=c*mlp
    h = c * mlp
    gdfn = c * (2 * h) + (2 * h) * 9 + h * c
    return blocks * (mdta + gdfn)


def restormer_params(channels, num_blocks, num_refinement, mlp):
    L = len(channels)
    p = 0
    p += 2 * channels[0] * 9          # embed_conv 3x3 (in=2)
    for i in range(L):
        p += restormer_level_params(channels[i], num_blocks[i], mlp)   # encoders
    for i in range(L - 1):
        p += channels[i] * (channels[i] // 2) * 9      # down conv 3x3
        p += channels[i + 1] * (channels[i + 1] * 2) * 9   # up conv 3x3
        p += 2 * channels[i] * channels[i]             # reduce 1x1
        p += restormer_level_params(channels[i], num_blocks[i], mlp)   # decoders
    p += restormer_level_params(channels[0], num_refinement, mlp)      # refinement
    p += channels[0] * 2 * 9          # output 3x3
    return p


CSME_PARAMS = 1_500_000   # ComplexUnet_DCAE {12,24,48,96,192} 고정, 대략치


def total_params(cascades, channels, num_blocks, num_refinement, mlp):
    per = restormer_params(channels, num_blocks, num_refinement, mlp) + CSME_PARAMS
    return cascades * per


def activation_score(cascades, channels, H=384, W=372):
    # 상대 점수: Σ(채널 × 공간) × cascade (checkpointing으로 실제론 완화됨)
    L = len(channels)
    s = 0
    for i in range(L):
        s += channels[i] * (H // (2 ** i)) * (W // (2 ** i))
    return cascades * s


if __name__ == "__main__":
    num_blocks = [3, 6]
    num_refinement = 2
    mlp = 3
    cascade_grid = [2, 4, 6, 8]
    channel_grid = [[16, 32], [24, 48], [32, 64]]
    print("== 근사 파라미터 수 (M) — 절대치 아님, 상대 비교용 ==")
    print(f"{'T':>3} | " + " | ".join(f"{str(c):>9}" for c in channel_grid))
    for T in cascade_grid:
        row = [f"{total_params(T, ch, num_blocks, num_refinement, mlp)/1e6:>7.1f}M" for ch in channel_grid]
        print(f"{T:>3} | " + " | ".join(f"{r:>9}" for r in row))
    print("\n== 상대 활성값-메모리 점수 (클수록 VRAM 위험) ==")
    for T in cascade_grid:
        row = [f"{activation_score(T, ch)/1e6:>7.0f}" for ch in channel_grid]
        print(f"{T:>3} | " + " | ".join(f"{r:>9}" for r in row))
```

- [ ] **스텝 2: 실행 + AST 파싱 확인**

Run: `python scripts/estimate_sdum_analytical.py`
기대: 두 개의 4×3 표 출력(파라미터 M, 상대 메모리 점수). torch 없이 돎.
Run: `python -c "import ast; ast.parse(open('scripts/estimate_sdum_analytical.py',encoding='utf-8').read()); print('ok')"`

- [ ] **스텝 3: 커밋**

```bash
git add scripts/estimate_sdum_analytical.py
git commit -m "feat: analytical (no-torch) SDUM size/memory ranker to narrow 8GB sweep"
```

---

## Task C2: 정밀 파라미터 실측 (선택 — venv 있을 때만)

> C1은 근사. 정확한 파라미터 수가 필요하면(그리고 CPU venv를 만들 의향이 있으면) 모델을 빌드해 실측. **forward는 안 함**(빌드+카운트만) → CPU RAM 부담 작음. venv 없으면 이 태스크는 건너뛴다.

**파일:**
- Create: `scripts/estimate_sdum.py`
- (필요 시) Modify: `.gitignore` (`.venv/`)

**인터페이스:**
- 소비: `utils.model.sdum.config.tiny_8gb_args`, `build_cascaded`. (torch/MONAI 필요)
- 산출: 정밀 파라미터 수 표.

- [ ] **스텝 1: (venv 없으면) CPU venv 만들고 설치**

```bash
python -m venv .venv
source .venv/Scripts/activate            # PowerShell: .venv\Scripts\Activate.ps1
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install monai einops timm numpy
echo ".venv/" >> .gitignore
```

- [ ] **스텝 2: 실측 스크립트 작성**

`scripts/estimate_sdum.py` 생성:

```python
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.model.sdum.config import tiny_8gb_args, build_cascaded


def count(cascades, channels):
    m = build_cascaded(tiny_8gb_args(num_cascades=cascades, channels=channels))
    return sum(p.numel() for p in m.parameters())


if __name__ == "__main__":
    for T in [2, 4, 6, 8]:
        row = [f"{count(T, ch)/1e6:>7.1f}M" for ch in ([16, 32], [24, 48], [32, 64])]
        print(f"T={T:>2}: " + "  ".join(row))
```

- [ ] **스텝 3: 실행 (venv 활성 상태)**

Run: `python scripts/estimate_sdum.py`
기대: 정밀 파라미터 표. C1 근사값과 비교해 근사식이 크게 틀리지 않았는지 확인(같은 자릿수면 OK).

- [ ] **스텝 4: 커밋**

```bash
git add scripts/estimate_sdum.py .gitignore
git commit -m "feat: exact SDUM param-count script (CPU venv, build-only)"
```

---

## Task C3: bbox 가중 손실 스캐폴딩 (작성만, 무설치)

> 순위축이 SSIM_bbox이므로 병변영역 가중 손실을 **미리 작성**해 둔다. 지금은 **코드 작성 + AST 검증**까지(실행·데이터 검증은 나중). 실제 활성화엔 dataloader가 bbox를 emit해야 함(범위 밖).

**파일:**
- Modify: `utils/common/loss_function.py`

**인터페이스:**
- 소비: 기존 `utils/common/loss_function.py`의 SSIM 손실, `utils/common/metrics.py`의 SSIM map.
- 산출: `SSIMBboxLoss` 클래스 — `forward(output, target, data_range, bboxes=None, lam=1.0)`. bbox 없으면 기존 SSIM 손실과 동일(하위호환).

- [ ] **스텝 1: 기존 손실/지표 코드 정독**

`utils/common/loss_function.py`(기존 SSIM 손실 구조)와 `utils/common/metrics.py`(`SSIM.forward`는 per-pixel map 반환, `foreground_mask`, `ssim_bbox`)를 읽어 시그니처·반환형을 확인. 새 클래스는 기존 SSIMLoss의 스타일을 그대로 따른다.

- [ ] **스텝 2: `SSIMBboxLoss` 추가**

`utils/common/loss_function.py` 하단에 추가(기존 SSIMLoss가 win_size 등 어떤 필드를 쓰는지 스텝1에서 확인해 맞춤; 아래는 metrics.SSIM 재사용 형태):

```python
class SSIMBboxLoss(nn.Module):
    """L = (1 - SSIM_full) + lam * (1 - SSIM_bbox).

    bboxes=None 이면 bbox 항 생략 → 기존 full-SSIM 손실과 동일(하위호환).
    실제 사용하려면 dataloader가 슬라이스별 bbox 리스트를 넘겨야 함(현재 미배선).
    """

    def __init__(self, win_size: int = 7, k1: float = 0.01, k2: float = 0.03):
        super().__init__()
        from utils.common.metrics import SSIM
        self.ssim = SSIM(win_size=win_size, k1=k1, k2=k2)
        self.win = win_size

    def _ssim_mean(self, recon, target, data_range):
        m = self.ssim(recon, target, data_range)   # per-pixel map
        return m.mean()

    def forward(self, output, target, data_range, bboxes=None, lam: float = 1.0):
        # output/target: (B,H,W) 또는 (H,W). 배치 평균.
        if output.dim() == 2:
            output = output.unsqueeze(0)
            target = target.unsqueeze(0)
        full = 0.0
        for b in range(output.shape[0]):
            full = full + (1.0 - self._ssim_mean(output[b], target[b], data_range))
        loss = full / output.shape[0]
        if bboxes:
            bt = 0.0
            n = 0
            for b, boxes in enumerate(bboxes):
                for box in (boxes or []):
                    x0, y0 = max(0, box["x"]), max(0, box["y"])
                    x1 = min(output.shape[-1], box["x"] + box["width"])
                    y1 = min(output.shape[-2], box["y"] + box["height"])
                    if (x1 - x0) < self.win or (y1 - y0) < self.win:
                        continue
                    rc = output[b, y0:y1, x0:x1]
                    tg = target[b, y0:y1, x0:x1]
                    bt = bt + (1.0 - self._ssim_mean(rc, tg, data_range))
                    n += 1
            if n > 0:
                loss = loss + lam * (bt / n)
        return loss
```

`import torch.nn as nn`이 파일 상단에 없으면 추가.

- [ ] **스텝 3: AST 파싱 확인**

Run: `python -c "import ast; ast.parse(open('utils/common/loss_function.py',encoding='utf-8').read()); print('ok')"`
(실행/수치 검증은 venv 또는 GPU 환경에서 나중에.)

- [ ] **스텝 4: 커밋**

```bash
git add utils/common/loss_function.py
git commit -m "feat(loss): SSIMBboxLoss scaffold (bbox-optional, backward-compatible)"
```

---

## Task C4: Vendored 코드 리뷰 + watch-list (무설치)

> GPU 첫 실행 전에 vendored SDUM에서 knee/2D·8GB에 안 맞는 하드코딩/가정을 훑어 정리. 정독 위주, 사소 수정만.

**파일:**
- Create: `docs/sdum_review_notes.md`
- (선택) Modify: `utils/model/sdum/*.py` (안전한 사소 수정만)

**인터페이스:**
- 소비: 없음(정독).
- 산출: 리뷰 노트 + GPU 첫 실행 watch-list.

- [ ] **스텝 1: vendored 코드 훑기**

`utils/model/sdum/restormer.py`, `nv_varnet.py`, `adapter.py`를 읽고 아래를 노트에 정리:
- `restormer_mri`의 `self.num_coils = 10` 하드코딩(우리 15코일; `use_csm=True`라 미사용).
- `soft_dc` dc_weight_map `(768,768)` knee 분기 — 우리 k-space보다 크니 center-crop(정상).
- `soft_dc`가 FP32 강제(`autocast(enabled=False)`) — 1080 FP32와 정합.
- 어댑터의 mask broadcast / 출력 도메인 = GPU 첫 실행 최우선 확인 지점.

- [ ] **스텝 2: watch-list 문서 작성**

`docs/sdum_review_notes.md` 생성. 위 항목 + "GPU 첫 실행에서 볼 것"(shape 에러 후보, MONAI 버전, dtype)을 목록화. (텐서 shape 흐름 상세가 필요하면 이 문서에 함께 정리.)

- [ ] **스텝 3: 커밋**

```bash
git add docs/sdum_review_notes.md
git commit -m "docs: SDUM vendored-code review notes + GPU-first-run watch-list"
```

---

## 범위 밖 (별도/나중)

- **옵션 A: CPU에서 pytest로 어댑터 검증** — 이 플랜엔 없음(사용자가 C를 택함). 원하면 별도로 진행 가능(가장 확실한 무-GPU 검증이지만 설치 필요).
- 실제 8GB 스윕·본 학습(GPU), GRAPPA prior(타팀), bbox 손실 **데이터 플러밍**, CSME 옵션(c) 주기적 재추정.

## Self-Review

- **스펙(C) 커버리지:** C의 4항목 매핑 — 분석적 VRAM/param 추정→C1, 파라미터 실측("설치되면")→C2(선택), 스캐폴딩→C3(bbox; GRAPPA는 타팀 제외, CSME(c)는 범위 밖), 코드 리뷰·정리→C4. ✅ 옵션 A(검증)는 명시적으로 제외.
- **Placeholder 점검:** C1/C2/C3에 실제 실행 코드 포함. C3는 "기존 코드 정독 후 스타일 맞춤" 스텝이 있으나 실제 클래스 코드도 제공(placeholder 아님).
- **타입 일관성:** `tiny_8gb_args`/`build_cascaded`는 기존 SDUM 코드와 일치. C1은 순수 함수(torch 무관), C2는 build_cascaded 사용.
- **정확성 주의:** C1 수치는 근사임을 문서·주석·출력 헤더에 명시. 정밀은 C2/GPU에서.
