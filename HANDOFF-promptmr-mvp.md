# HANDOFF — PromptMR+ MVP (GPU 검증 대기)

> GPU 서버 세션(Claude 포함)이 **먼저 읽는** 핸드오프. PromptMR+ MVP 코드는 로컬에서 구현·리뷰 완료됐고, 이제 **서버에서 GPU 검증만** 남았다. **검증 통과 후 이 파일은 삭제**한다.
>
> - 날짜: 2026-07-19 · 브랜치: `promptMR_plus` (master 미머지) · 상태: 코드 done, GPU 검증 pending
> - 근거 문서: `docs/superpowers/specs/2026-07-19-promptmr-plus-mvp-design.md` (설계), `docs/superpowers/plans/2026-07-19-promptmr-plus-mvp.md` (구현 플랜)

---

## 1. 지금까지 (코드 — 전부 리뷰 clean)

`promptMR_plus` 브랜치, 커밋 7개:

```
01f606e  GPU 검증 슬롯 shell11~13 (probe/run1 smoke/recon_eval)
61c8e0f  --model promptmr 배선 (train/test/reconstruct + AdamW)
57af0aa  build_promptmr 어댑터
cad29ab  tiny config factory (8GB·full "+" ON)
5944205  PromptMR+ backbone vendoring (clone base, v2 정합·SNU mask 수정)
02262c5  reentrant checkpointing 래퍼
5f74c2e  fastmri sens_expand/sens_reduce
```

- **모델**: `--model promptmr` (기존 `--model varnet` 경로는 그대로). backbone `utils/model/promptmr/promptmr.py`(vendored, 하이브리드: clone base + 공식 v2 정합), `config.py`(tiny config, "+" 4개 전부 ON, `num_adj_slices=1`), `adapter.py`(`build_promptmr`).
- **의존**: `utils/model/fastmri/`에 `sens_expand`/`sens_reduce` 추가됨, `utils/model/reentrant_wrapper.py` 추가됨.
- **contract**: `forward(masked_kspace (B,C,H,W,2), mask (B,1,1,W,1)) -> (B,384,384)`. 고정 `recon_eval.py`는 무수정 — `test_part.load_model`이 체크포인트의 `saved.model`로 PromptMR을 자동 복원.
- **tiny config**(spec §3): `num_cascades=6, n_feat0=16, feature_dim=[24,32,40]`, prompt/sens 소형, `adaptive_input=T·n_buffer=4·n_history=3·use_sens_adj=T`(풀 "+"), `learnable_prompt=T`, gradient checkpointing ON.

---

## 2. 지금 할 것 — GPU 검증 (기존 SDUM 큐는 건드리지 말 것)

새 슬롯 3개(`shell11~13`)만 실행. **`shell1~10`·`shellB`는 SDUM 큐 — 손대지 않는다.**

```bash
bash shell_scripting/shell11.sh   # probe
bash shell_scripting/shell12.sh   # run1 smoke
bash shell_scripting/shell13.sh   # recon_eval 채점
```

| 슬롯 | 무엇 | 통과 기준 (로그에서 확인) |
|---|---|---|
| `shell11` | build/shape/OOM probe (`num_cascades=2`, `--max-iters 2`) | OOM 없음 · 출력 shape `(1,384,384)` · **첫 val SSIM ≠ 0** · peak CUDA mem |
| `shell12` | run1 smoke (tiny config, 1 epoch) | train **loss 감소** · `../result/promptmr_run1/checkpoints/best_model.pt` 저장 |
| `shell13` | 공식 채점 | `SSIM_full` / `SSIM_bbox` / 추론시간 3축 산출 |

- **shell11 OOM이면** — 레버 순서(spec §6): `--readout-crop`(이미 ON) → coil-compress(`transforms.py` 기존 레버) → `--num-cascades` 6→4 → `--n-feat0` 16→8. **큐 구조·config 크게 바꾸기 전 사용자에게 확인**(CLAUDE.md '파괴적 행동').
- **shell12 1 epoch이 너무 길면** — 명령에 `--max-iters 300 -r 50` 추가(슬롯 주석에 안내). run 번호(run1)는 유지.
- **resume 확인**하려면 shell12를 `--resume` 붙여 재실행(같은 config).

---

## 3. 주의사항 (실행 중·해석 시)

1. **스케일 정합 = 첫 관문.** PromptMR+는 k-space native라 내부에서 정규화(`NormPromptUnet` norm→net→denorm)한다. 데이터로더 `--normalize`는 **쓰지 않는다**(이중 정규화 방지). probe에서 **SSIM이 ~0으로 나오면** 스케일 정합 실패 신호 → denorm/타깃 스케일 먼저 의심.
2. **`center_crop`이 `cuda:0` 하드코딩** (`utils/common/utils.py`). VarNet과 공유하는 **기존** 이슈. → **`-g 0`(GPU_NUM=0)으로만 돌릴 것.** GPU_NUM≠0이면 device mismatch. MVP 수정 대상 아님.
3. **속도도 점수 축.** `recon_eval.py`는 volume forward가 60s 넘으면 경고. PromptMR+는 `compute_sens_per_coil=True`(per-coil 루프, 메모리 절약) + unrolled + checkpoint recompute + FP32라 느릴 수 있다. 시간 로그 주시. (튜닝 사안이지 버그 아님.)
4. **mask 처리는 검증된 패턴.** vendored `forward()`에서 clone의 SNU-align 블록(`.cuda()`+reshape)을 제거하고 `(B,1,1,W,1)` mask를 그대로 넘긴다(`num_low_frequencies=None`→ACS 자동 도출). 이는 **배포된 VarNet이 같은 mask에 쓰는 `mask[:,0,0,:,0]`·`torch.where(mask,…)`와 동일 패턴** — 리뷰에서 확인됨. shape 에러 나면 여기부터 볼 것은 아니고, 오히려 coil/H broadcast를 먼저 확인.
5. **FP32 전용** (Pascal). AMP/BF16 절대 금지 — 코드에 없음, 추가하지 말 것.
6. **`num_adj_slices=1`** (단일 슬라이스). adjacent-slice 2.5D는 후속 increment(데이터로더 변경 필요). `learnable_prompt=True`는 의도적(충실도).

---

## 4. 검증 결과 기록

3축 수치를 `training_anal/`에 새 md로 기록(핵심요약/ML 체크리스트 규약), **VarNet baseline(run11: full 0.9118 / bbox 0.9021 등) 대비**로. run 넘버링·`-r 100` 규칙은 `shell_scripting/README.md` 참조. 이번 MVP 목표는 "VarNet을 이긴다"가 **아니라** "8GB에서 학습되고 채점된다" — 첫 숫자 확보가 성공.

---

## 5. 검증 통과 후

- `git push origin promptMR_plus` (이 서버는 **PAT 필요** — CLAUDE.md §9).
- **그 다음에** 브랜치 마무리(master 머지 여부) 결정. 검증 전 master 머지 금지.
- 스케일업 increment(채널·cascade↑, adjacent-slice, coil-compress 튜닝, fg-loss/bbox 학습)는 각각 별도 spec→plan (spec §8).

---

## 6. 커밋 규약 (재확인)

- 커밋 전 **누가 커밋하는지** 먼저 확인 → `git config user.{name,email}`(local). 현재 local = `kindofwater`.
- **AI/Claude attribution 금지** (커밋·PR 어디에도).
