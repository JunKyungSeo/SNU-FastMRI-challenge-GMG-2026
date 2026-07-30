# SDUM 학습 문제 수정 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** run1에서 드러난 두 문제 — (P0) 손실이 SSIM≈0에 고정돼 학습이 안 됨, (P1) 반복(iteration)이 느리고 디버깅 사이클이 9시간 — 를 고쳐서 "실제로 배우는" 짧은 학습을 가능하게 한다.

**Architecture:** 원인은 어댑터가 `ifft(masked_kspace)`를 **정규화 없이** 모델에 넣는 것(스케일이 타깃과 안 맞음 + 신경망이 비-단위 스케일 입력에서 못 배움). 어댑터에 **per-sample 스케일 정규화(입력 나누고 출력 곱하기)** 를 넣어 scale-equivariant하게 만들면 SSIM 학습이 성립한다. 속도는 `num_workers`(현재 0)와 빠른 디버깅용 `--max-iters`로 사이클을 줄인다. LR warmup은 transformer scratch 안정화(선택).

**Tech Stack:** PyTorch, MONAI(≥1.3, `apps.reconstruction`), NV SDUM cascaded 모델(vendored), pytest.

## Global Constraints

- **하드웨어:** GTX 1080 **8GB, FP32 전용**(Pascal → AMP/BF16 금지). batch=1 고정.
- **데이터:** knee 전용, Cartesian acc4/acc8, 출력 **384×384**.
- **규정:** 사전학습 가중치/외부데이터 금지 → HuggingFace 자동다운로드 vendor 안 함(유지).
- **호환성:** 기존 VarNet 경로 깨지지 말 것(`--model` 기본값 `varnet`). 수정은 **SDUM 경로에만** 적용.
- **실행 환경:** 로컬엔 torch 없음 → **모든 실제 실행/테스트는 GPU 서버 또는 Colab**. Task 2의 scale-equivariance 테스트는 합성 데이터라 **Colab CPU에서도** 돈다.
- **손상된 체크포인트:** 기존 sdum_run1 체크포인트(loss≈1.0)는 정규화 도입으로 무의미 → 버리고 새로 학습.

---

## File Structure

| 파일 | 역할 | 변경 |
|---|---|---|
| `utils/model/sdum/adapter.py` | k-space↔영상 어댑터 | **수정** — forward에 per-sample 스케일 정규화 추가 |
| `tests/test_sdum_adapter.py` | 어댑터 테스트 | **수정** — scale-equivariance 테스트 추가 |
| `utils/learning/train_part.py` | 학습 루프 | **수정** — `--max-iters` early-break, (선택) LR scheduler 배선 |
| `train.py` | 진입점/인자 | **수정** — `--num-workers`, `--max-iters`, (선택) `--warmup-iters` 추가 |

정규화 로직은 **어댑터 안에만** 둔다(모델 계약 `forward(kspace,mask)->(B,384,384)` 불변, 학습/추론 양쪽 자동 적용).

---

## Task 1: 스케일 불일치 진단 (fix 전/후 확인용, P0)

수정 전에 "정말 스케일이 원인인지" 숫자로 확인하고, 수정 후 "고쳐졌는지" 같은 방법으로 검증한다. 코드 변경 없이 **일회성 진단**.

**Files:**
- (임시) `utils/learning/train_part.py:30-31` 사이에 debug print 삽입 후 제거

**Interfaces:**
- Consumes: `output=model(kspace,mask)` (B,384,384), `target` (B,384,384), `maximum` (B,)
- Produces: 콘솔에 스케일 비율 (진단값, 커밋 안 함)

- [ ] **Step 1: 임시 진단 print 삽입**

`train_epoch`의 `output = model(kspace, mask)` 바로 다음에 임시로:

```python
        output = model(kspace, mask)
        if iter == 0:
            print(f"[SCALE] out mean={output.mean():.3e} max={output.max():.3e} | "
                  f"tgt mean={target.mean():.3e} max={target.max():.3e} | "
                  f"attrs_max={maximum.max():.3e}", flush=True)
```

- [ ] **Step 2: 1 iter만 돌려 스케일 확인**

Run (GPU 서버):
```bash
python train.py --model sdum --num-cascades 4 --channels 16 32 \
  -b 1 -e 1 --max-iters 1 -n scale_probe -t /root/Data/train/ -v /root/Data/val/
```
(`--max-iters`는 Task 4에서 추가됨. 먼저 확인만 하려면 첫 로그 뜨자마자 Ctrl-C.)
Expected: `[SCALE] out mean=... tgt mean=...` 출력. **out과 tgt의 크기 차이(배수)를 기록.** 수십~수백 배 이상 벌어져 있으면 스케일이 원인 확정.

- [ ] **Step 3: 진단 print 제거 (커밋하지 않음)**

Step 1에서 넣은 3줄 삭제. 이 Task는 커밋 없음(진단 전용).

---

## Task 2: 어댑터 per-sample 스케일 정규화 (P0 — 핵심 수정)

입력 영상을 per-sample 스케일 `s`로 나눠 신경망에 ~단위 스케일로 주고, 출력 magnitude에 `s`를 다시 곱해 타깃 물리 스케일로 되돌린다. 모델 내부 DC도 정규화된 k-space에 대해 일관되게 동작(선형)하므로 자기일관적이다.

**Files:**
- Modify: `utils/model/sdum/adapter.py:32-50` (forward 전체 교체)
- Test: `tests/test_sdum_adapter.py` (scale-equivariance 테스트 추가)

**Interfaces:**
- Consumes: `masked_kspace` (B,C,H,W,2), `mask` (B,1,1,W,1)
- Produces: `forward(...) -> (B,384,384)`; **불변식**: `forward(α·k, mask) == α·forward(k, mask)` (eval 모드, float 오차 내). 이 scale-equivariance가 SSIM 학습을 데이터 절대 스케일과 무관하게 만든다.

- [ ] **Step 1: scale-equivariance 실패 테스트 작성**

`tests/test_sdum_adapter.py`에 추가(파일 상단 `DEV`, `tiny_8gb_args`, `build_sdum`은 기존 것 재사용):

```python
def test_scale_equivariance():
    """forward(a*k) == a*forward(k): 정규화가 있어야 성립. 없으면 신경망 bias 때문에 깨짐."""
    torch.manual_seed(0)
    model = build_sdum(tiny_8gb_args(num_cascades=2, channels=[16, 32])).to(DEV).eval()
    B, C, H, W = 1, 4, 64, 40
    ksp = torch.randn(B, C, H, W, 2, device=DEV)
    mask = torch.zeros(B, 1, 1, W, 1, dtype=torch.uint8, device=DEV)
    mask[..., ::4, :] = 1
    mask[..., W // 2 - 6 : W // 2 + 6, :] = 1
    with torch.no_grad():
        o1 = model(ksp, mask)
        o10 = model(10.0 * ksp, mask)
    assert torch.allclose(o10, 10.0 * o1, rtol=1e-3, atol=1e-4)
```

- [ ] **Step 2: 테스트 실패 확인 (현재 어댑터엔 정규화 없음)**

Run (Colab/GPU): `pytest tests/test_sdum_adapter.py::test_scale_equivariance -v`
Expected: **FAIL** — `o10`이 `10*o1`과 다름(정규화 없어 신경망 bias가 스케일에 비선형 반응).

- [ ] **Step 3: 어댑터 forward에 정규화 구현**

`utils/model/sdum/adapter.py`의 `forward`를 다음으로 교체:

```python
    def forward(self, masked_kspace: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        B, C, H, W, two = masked_kspace.shape
        # 언더샘플 영상 = ifft(masked_kspace)  -> (B,C,H,W,2)
        img = ifftn_centered(masked_kspace, spatial_dims=2, is_complex=True)

        # per-sample 스케일: zero-filled RSS magnitude의 표준편차. 신경망에 ~단위 스케일 입력.
        zf_mag = root_sum_of_squares_t(complex_abs_t(img), spatial_dim=1)  # (B,H,W)
        scale = zf_mag.reshape(B, -1).std(dim=1).clamp_min(1e-12)          # (B,)
        img = img / scale.view(B, 1, 1, 1, 1)                              # broadcast over C,H,W,2

        x = img.unsqueeze(1)                      # (B,1,C,H,W,2)
        mask_b = mask.bool().view(B, 1, 1, 1, W, 1).expand(B, 1, C, H, W, two)
        acc = _extract_acc(mask.reshape(B, W)[0])

        out = self.cascaded(x, mask_b, self.mask_type, acc, self.acq_type)
        out = out[:, 0]                           # (B,C,H,W,2)
        mag = root_sum_of_squares_t(complex_abs_t(out), spatial_dim=1)  # (B,H,W)
        mag = mag * scale.view(B, 1, 1)           # 출력 스케일 복원 -> 타깃 물리 스케일

        return center_crop(mag, *self.out_size).to(masked_kspace.device)
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `pytest tests/test_sdum_adapter.py -v`
Expected: `test_scale_equivariance` PASS + 기존 `test_adapter_forward_shape_and_finite`, `test_one_training_step` PASS.

- [ ] **Step 5: (GPU) 실제 데이터로 loss 하강 확인 — 이 Task의 진짜 게이트**

Run (GPU 서버, subset·짧게 — Task 3·4 적용 후):
```bash
python train.py --model sdum --num-cascades 4 --channels 16 32 \
  -b 1 -l 2e-4 -e 1 --max-iters 200 -r 10 --num-workers 4 \
  -n sdum_scalefix -t /root/Data/train/ -v /root/Data/val/
```
Expected: **Loss가 1.0 근처에서 눈에 띄게 하강**(예: 200 iter 내 0.9→0.7대). Task 1 진단을 다시 켜보면 `out`과 `tgt` 스케일이 같은 자릿수. → 하강하면 P0 해결. **하강 안 하면** 스케일 외 구조 문제 → systematic-debugging으로 별도 조사(정규화만으론 부족).

- [ ] **Step 6: 커밋**

```bash
git add utils/model/sdum/adapter.py tests/test_sdum_adapter.py
git commit -m "fix(sdum): per-sample scale normalization in adapter (SSIM was ~0)"
```

---

## Task 3: `--num-workers` 배선 (P1 — 데이터 로딩 병렬화, 공짜 속도)

`load_data.py`는 이미 `getattr(args,"num_workers",0)`를 읽는다([load_data.py:206](utils/data/load_data.py#L206)). train.py에 인자만 추가하면 GPU가 데이터 대기로 노는 시간이 준다.

**Files:**
- Modify: `train.py:37` (인자 추가)

**Interfaces:**
- Produces: `args.num_workers` (int, 기본 4) → `create_data_loaders`가 소비

- [ ] **Step 1: 인자 추가**

`train.py`의 `--seed` 인자 다음 줄에 추가:

```python
    parser.add_argument('--num-workers', type=int, default=4, help='DataLoader worker processes (0=main proc)')
```

- [ ] **Step 2: 동작 확인**

Run (GPU): 아무 학습 명령에 `--num-workers 4` 붙여 실행 → 첫 로그의 `Time`(50 iter)이 이전(≈348s) 대비 줄었는지 비교.
Expected: iter 시간 감소(데이터 로딩이 연산과 겹침). 감소폭은 데이터 I/O 비중에 따라 다름.

- [ ] **Step 3: 커밋**

```bash
git add train.py
git commit -m "feat: --num-workers for parallel data loading (default 4)"
```

---

## Task 4: `--max-iters` 빠른 디버깅 (P1 — 9시간 epoch 회피)

전체 4651 iter 대신 앞부분 N iter만 돌려 "배우는지"를 몇 분 내 확인. 학습 루프에 early-break만 추가(데이터 로딩 불변).

**Files:**
- Modify: `train.py:37` (인자 추가), `utils/learning/train_part.py:23` (루프 break)

**Interfaces:**
- Produces: `args.max_iters` (int, 기본 0=제한 없음) → `train_epoch`가 소비

- [ ] **Step 1: 인자 추가**

`train.py`에 추가:
```python
    parser.add_argument('--max-iters', type=int, default=0, help='Cap train iters/epoch for debugging (0=full)')
```

- [ ] **Step 2: 루프 early-break 추가**

`utils/learning/train_part.py`의 `train_epoch` for 루프 맨 앞에:
```python
    for iter, data in enumerate(data_loader):
        if getattr(args, "max_iters", 0) and iter >= args.max_iters:
            break
        mask, kspace, target, maximum, _, _ = data
```

- [ ] **Step 3: 동작 확인**

Run (GPU): `python train.py --model sdum ... --max-iters 20 -r 10 -e 1 -n debug20 ...`
Expected: train이 20 iter에서 멈추고 val로 넘어감(전체 epoch 대기 없음).

- [ ] **Step 4: 커밋**

```bash
git add train.py utils/learning/train_part.py
git commit -m "feat: --max-iters to cap train iters for fast debug cycles"
```

---

## Task 5: LR warmup + cosine 스케줄 (P2 — 선택, transformer scratch 안정화)

Task 2로 loss가 하강하기 시작하면 대개 충분하다. warmup은 초기 발산이 보일 때만 적용. per-iter 스텝이라 스케줄러를 `train_epoch`로 전달.

**Files:**
- Modify: `train.py:37` (인자), `utils/learning/train_part.py` (scheduler 생성·전달·step)

**Interfaces:**
- Consumes: `args.warmup_iters`(int, 기본 0=off), `args.lr`, `args.num_epochs`, `len(train_loader)`
- Produces: per-iter로 `scheduler.step()` 호출; warmup_iters=0이면 스케줄러 미사용(기존과 동일)

- [ ] **Step 1: 인자 추가 (train.py)**

```python
    parser.add_argument('--warmup-iters', type=int, default=0, help='SDUM: linear LR warmup iters then cosine (0=off)')
```

- [ ] **Step 2: 스케줄러 생성 (train_part.py `train` 함수, optimizer 생성 직후)**

`import math`를 파일 상단에 추가하고, `train_loader` 생성 이후에:
```python
    scheduler = None
    if getattr(args, "model", "varnet") == "sdum" and getattr(args, "warmup_iters", 0) > 0:
        total_iters = args.num_epochs * len(train_loader)
        warmup = args.warmup_iters
        def lr_lambda(step):
            if step < warmup:
                return (step + 1) / warmup
            progress = (step - warmup) / max(1, total_iters - warmup)
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
```

- [ ] **Step 3: train_epoch에 scheduler 전달·step**

`train_epoch` 시그니처에 `scheduler=None` 추가, `optimizer.step()` 다음에:
```python
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
```
그리고 `train` 안 호출부를 `train_epoch(args, epoch, model, train_loader, optimizer, loss_type, scheduler)`로 수정.

- [ ] **Step 4: 동작 확인**

Run (GPU): `python train.py --model sdum ... --warmup-iters 100 --max-iters 200 -r 10 ...`
Expected: 초기 iter의 유효 lr가 작게 시작(발산 없이 loss 하강). warmup 없이도 하강하면 이 Task는 생략 가능.

- [ ] **Step 5: 커밋**

```bash
git add train.py utils/learning/train_part.py
git commit -m "feat(sdum): optional LR warmup+cosine schedule (--warmup-iters)"
```

---

## 실행 순서 권장

1. **Task 3 + Task 4 먼저**(빠른 사이클 확보) → 2. **Task 2**(정규화) 적용 후 **Step 5 게이트**(subset에서 loss 하강 확인)가 이 계획의 성패. → 3. 하강 확인되면 Task 1 진단으로 스케일 정합 재확인. → 4. 필요 시 Task 5(warmup). → 5. 그 다음에야 epoch/시간 예산과 config(cascade/channels) 스윕.

## Self-Review

- **스펙 커버리지:** run1 분석의 3문제 → P0 손실(Task 1·2), 속도 num_workers(Task 3)·9h epoch(Task 4), 하이퍼파라 warmup(Task 5). ✅ 모두 태스크 존재.
- **placeholder 스캔:** 모든 코드 스텝에 실제 코드 포함, "적절히 처리" 류 없음. ✅
- **타입 일관성:** `scale`는 (B,)→`view(B,1,1,1,1)`(영상 나눗셈)/`view(B,1,1)`(mag 곱). forward 반환 (B,384,384) 불변. `args.max_iters`/`num_workers`/`warmup_iters` 기본값이 기존 동작 보존(0/4/0). ✅
- **미확정 리스크(정직):** Task 2 Step 5에서 loss가 여전히 안 떨어지면 원인은 스케일이 아니라 구조(어댑터 mask broadcast/DC/coil-combine) → CLAUDE.md §2 "불확실 포인트" + systematic-debugging으로 escalation. 이 계획은 "가장 유력한 원인(스케일)"을 먼저 확정적으로 배제/수정하는 순서로 설계됨.
