# Resume Training (이어 학습) — 설계 스펙 (2026-07-09)

SDUM 학습을 checkpoint에서 이어서 재개하는 기능. run7 분석의 ML 체크리스트에서 "⑦ resume 없음" 갭으로 식별됨.

## 목표 (use cases)
1. **중단(crash/kill)된 run을 원래 계획 epoch까지 이어감** — 긴 학습(run당 ~10–20h)이 중간에 죽어도 처음부터 안 함.
2. **학습을 중간에 끊고 `recon_eval`로 평가한 뒤 다시 이어가는 loop** — 진행 상태를 중간 채점해보고 계속.

## 확정 결정 (brainstorming)
- **단위 = epoch 경계.** `save_model`이 이미 매 epoch 끝에 저장 → 그 지점에서 재개. mid-epoch 정확 재개는 안 함(dataloader 위치·RNG 복원 회피 = 견고, "오류없이" 우선). 중단 시 진행 중이던 epoch은 버리고 **직전 완료 epoch**부터.
- **범위 = 원래 계획까지만.** cosine horizon은 원래 `num_epochs` 그대로 이어짐(scheduler를 그 지점까지 재현). "이미 끝난 run을 더 늘리기"(재-warmup·새 horizon)는 **범위 밖** — 그건 fresh run으로.

## 현재 상태 (근거)
- `save_model` (train_part.py:84-97): `epoch/args/model/optimizer/best_val_loss` → `model.pt`, `is_new_best`면 `best_model.pt`. **scheduler·RNG·val_loss_log 미저장.**
- `train()` (train_part.py:100-): `start_epoch=0`, `best_val_loss=1.` 하드코딩, load 없음. scheduler = `LambdaLR(total_iters=num_epochs×len_loader)`, per-iter step.
- ✅ **핵심 관찰**: fast-forward 방식(§3)을 쓰면 `save_model`이 저장하는 `epoch/args/optimizer/model/best_val_loss`만으로 **충분** → `save_model` 변경 불필요. resume는 순수 `train()` 추가.

## 설계 (approach A: `--resume` 플래그)

### 1. 트리거 & 선택
`train.py --resume`. `../result/<net_name>/checkpoints/`**`model.pt`**(=마지막 완료 epoch) 로드. **`best_model.pt` 아님**(best는 더 이른 epoch일 수 있어 재개엔 부적합). 플래그 있는데 파일 없으면 **즉시 에러 종료**(조용히 scratch 금지).

### 2. 복원 대상
- `model.load_state_dict(ckpt['model'])`, `optimizer.load_state_dict(ckpt['optimizer'])`(Adam 모멘트).
- `start_epoch = ckpt['epoch']`, `best_val_loss = ckpt['best_val_loss']`.
- epoch 루프 `range(start_epoch, args.num_epochs)`.

### 3. Scheduler 재개 (핵심 "오류없이" 지점)
직렬화하지 않고 **fast-forward**로 재현. 순서가 중요:
1. `optimizer = AdamW(base lr=args.lr)`.
2. `scheduler = LambdaLR(optimizer, lr_lambda)` — step0, 이때 `initial_lr=base`로 설정됨.
3. `optimizer.load_state_dict(ckpt['optimizer'])` — Adam 모멘트 복원.
4. `for _ in range(start_epoch × len(train_loader)): scheduler.step()` — cosine을 재개 지점까지 전진.

→ cosine이 정확히 재개 지점부터 이어짐. **base-lr 시점에 scheduler를 만들어** PyTorch의 `initial_lr` 미설정 함정을 회피(디케이된 lr로 scheduler를 만들면 base가 틀어짐). fast-forward 비용 = lambda 평가 수천 회(<1s).

> ⚠️ **fast-forward 카운트 정확성**: scheduler는 **train iter당 1회** step. `--max-iters N`이면 epoch당 step은 `len_loader`가 아니라 `min(N, len_loader)`다. 따라서 `steps_per_epoch = min(max_iters, len_loader) if max_iters>0 else len_loader`, fast-forward = `start_epoch × steps_per_epoch`. (LambdaLR 생성 시 내부적으로 step()이 1회 불려 last_epoch 0이 됨 = 무중단 run의 iter0 이전 상태와 동일하므로, 이후 fast-forward 횟수는 "재개까지 실제 돈 train iter 수"와 정확히 일치.)

**가드**: 저장 `args`의 `num_epochs`·`warmup_iters`가 현재와 다르면 → 에러(cosine horizon 어긋남). `len_loader`도 동일 전제(데이터·batch 동일). `warmup_iters=0`으로 학습된 checkpoint(scheduler=None)면 fast-forward 스킵(상수 LR).

### 4. val_loss_log 이어쓰기
`--resume`이면 기존 `val_loss_log_<run_tag>.npy` **로드 후 append**(덮어쓰기 금지). 없으면 빈 배열로 시작.

### 5. 재현성
시작 시 기존 `seed_fix` 재시드. resume 이어짐은 `(checkpoint, seed)`로 **결정론적**(같은 재개는 같은 결과). 단 무중단 run과 **byte-동일은 아님**(재개 epoch의 shuffle 순서가 다름) — 대회 재현성(스크래치+seed → 동일)엔 영향 없어 허용.

### 6. Config 불일치 방어
`--resume` 시 저장 `args`의 `model·num_cascades·channels`가 현재 인자와 다르면 **거부**(다른 아키텍처 checkpoint 이어받기 = 오류). 태그충돌(예: run7이 `sdum_run5`)은 `-n` 일치로 안전(같은 dir 로드).

### 7. 중단→eval→재개 워크플로우
학습 kill(큐 스킵/`pkill`) → 마지막 `model.pt`=직전 완료 epoch → `recon_eval.py -n <name> -p /root/Data/leaderboard` 채점 → `train.py --resume -n <name> ...`로 재개. `strategy/collaboration_workflow.md`에 절차 추가.

## 에러 처리
- `--resume` + checkpoint 없음 → 명확한 에러 메시지 후 종료.
- config/horizon 불일치 → 거부 + 무엇이 다른지 출력.
- scheduler=None(warmup 0) checkpoint → fast-forward 스킵(정상, 상수 LR).

## 테스트
- **왕복**: save→load 후 model·optimizer state 일치.
- **fast-forward lr**: 재개 후 optimizer lr == 무중단 run의 동일 step lr(수치 비교, 허용오차 내).
- **스모크(GPU, 큐)**: `-e 2 --max-iters N`로 epoch0 완료 후 kill → `--resume` → epoch1부터 이어지고 loss 유한·scheduler lr 연속인지.

## 손대는 파일 (예상)
- `train.py`: `--resume` 플래그(store_true).
- `utils/learning/train_part.py`: `train()`에 load·fast-forward·val_log append·가드. (`save_model`은 **불변**.)
- `strategy/collaboration_workflow.md`: 중단→eval→재개 절차.

## 범위 밖 (YAGNI)
- mid-epoch 정확 재개(dataloader 위치 복원).
- 끝난 run 연장(재-warmup·새 cosine horizon).
- 자동 resume(플래그 없이 checkpoint 감지).
- RNG 상태 직렬화(무중단 byte-동일).
