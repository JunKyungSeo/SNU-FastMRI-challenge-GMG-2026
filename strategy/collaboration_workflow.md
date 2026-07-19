# 협업 워크플로우 — train → 분석 → 코멘트 → 반영 → 반복

이 repo에서 **학습(현재 백본 = PromptMR+)을 굴리는 전체 사이클.** 사람(팀원)과 Claude가 번갈아 도는 루프다.

```
 ① 큐에 훈련 등록  →  ② 실행(로그 남김)  →  ③ Claude에 분석 요청
        ↑                                             │
        │                                             ▼
 ⑤ 큐/코드 수정  ←  ④ 분석 md에 코멘트(사람)  ←  training_anal/ 보고서
        └────────────────────── 반복 ──────────────────────┘
```

**대원칙:** GPU는 **큐 스크립트에서만**, 판단·기록은 **문서로** 남긴다.

---

## ① 훈련을 큐에 등록  (`shell_scripting/`)

GPU 쓰는 명령(train/recon)은 **`shell_scripting/shellN.sh` 슬롯**에만 넣는다.
가벼운 건(h5_viewer·pytest·git) 큐에 넣지 말고 **옆 분할 터미널에서 바로** 실행.

**슬롯 채우기** — 슬롯 파일을 열어 (a) 맨 위 `# 목적:` 한 줄, (b) 실제 명령 작성:
```bash
python train.py --model promptmr --num-cascades 10 --n-feat0 48 --sens-n-feat0 24 --use-checkpoint \
  -b 1 -l 1e-6 -e 10 -r 500 --fg-loss --area-weight --grad-clip 0.1 --accum-steps 2 \
  -n promptmr_run1 --run-tag run1 -t /root/Data/train/ -v /root/Data/val/
```
- `-n <name>` : 결과 폴더 `../result/<name>/`. **run마다 다르게.**
- `--run-tag runN` : `val_loss_log`을 run별로 분리 저장(덮어쓰기 방지).
- 새 슬롯이 더 필요하면 `cp shell_scripting/_template.sh shell_scripting/shell11.sh`.

**실행** — 채운 슬롯을 세미콜론으로 이어서(직렬 큐):
```bash
bash shell_scripting/shell1.sh ; bash shell_scripting/shell2.sh
```
- 로그: `shell_scripting/logs/<슬롯>_<timestamp>.log` (실행 명령까지 `set -x`로 기록됨).
- ⚠️ **9시간 걸기 전 스모크 테스트 먼저:** 같은 명령에 `--max-iters 30 -e 1 -r 5`를 붙여 몇 분 돌려 → loss 유한·스케줄러 동작·지표 로깅·실제 s/iter 확인.

## ② Claude에게 분석 요청

run이 끝나면(또는 중단하면):
> **"logs의 run5 분석해서 training_anal 보고서 만들어줘"**

Claude가 `training_anal/YYYY-MM-DD-run5-*-analysis.md`를 만든다. 작성 규칙은 **[`training_anal/CLAUDE.md`](../training_anal/CLAUDE.md)** — 반드시 ⓐ **직전 run 대비 무엇이 바뀌었나**, ⓑ **이전 분석의 지적이 이번에 어떻게 됐나**를 담고, 숫자는 로그로 검증한다.

## ③ 분석 md에 코멘트 달기 (사람)

분석 문서를 읽고 **맨 아래 `## 💬 코멘트`** 섹션에 결정을 적는다. 예:
> - run6은 셸스크립트 확인용이니 분석에서 빼라.
> - full SSIM을 먼저 정확히 잡는 게 1순위. bbox는 그다음.
> - §5-A는 알아서 수정 / §5-B는 run번호 붙여 저장하게 코드 수정 / §5-C는 보류.

## ④ 코멘트를 반영시키기 (Claude 실행)

> **"내가 단 코멘트를 프롬프트라 생각하고 실행해"**

Claude가 코멘트대로 **큐 슬롯 명령 · 코드 · 문서**를 수정한다.
(커밋은 사람이 **"커밋해"** 라고 할 때만. 보통 몇 개 모아 한 번에.)

## ⑤ 다음 run 큐 등록 → ①로 반복

수정된 설정으로 다음 슬롯을 채워 다시 ①.

---

## 뭘 바꿀지 모를 땐

Claude에게 근거 기반 추천을 요청:
> "다음 run에서 어떤 하이퍼파라미터/코드를 바꾸면 좋을지 코드 근거로 분석해줘"

→ 속도/용량/최적화/손실 레버를 코드로 확인해 **다음 run 명령 + 필요한 코드 변경**을 제안받는다.

---

## 고려할 점

python 파일을 수정해도 현재 수행중인 학습에 영향을 끼치지 않음
> 영향을 끼치는 것은 그 다음 학습부터.

소통하는 시가는 md에 코멘트 달면서 하는 게 좋음
> 코드 수정 하기 전에 한 번 대화하고 하면 좋겠죠? 
---

## 학습 중단 → 채점 → 재개 (resume)

긴 학습을 중간에 끊고 채점해보거나, 죽은 학습을 이어갈 때:

1. **중단**: 큐 스킵(다음 슬롯으로) 또는 `pkill -f train.py`. epoch 경계 저장이라 **직전 완료 epoch**의 `model.pt`가 남는다(진행 중이던 epoch은 버려짐).
2. **채점**: `python recon_eval.py -n <net_name> -p /root/Data/leaderboard` → SSIM_full/bbox/시간 3축.
3. **재개**: 같은 명령에 `--resume`만 추가해 다시 큐로. `-n`·config(`--num-cascades`/`--n-feat0`/`--sens-n-feat0`/`--n-history`/`--num-epochs`/`--warmup-iters`/`--lr`/`--accum-steps`)는 **원래와 동일해야** 함(다르면 거부됨). cosine은 원래 horizon 이어서 감쇠.

> resume는 "원래 계획 epoch까지" 이어감. 더 길게 하려면 fresh run으로 새로.

## 참고 파일
- **[`shell_scripting/README.md`](../shell_scripting/README.md)** · `_template.sh` — 큐 규칙·슬롯 원본
- **[`training_anal/CLAUDE.md`](../training_anal/CLAUDE.md)** — 분석 작성 규칙 + run 이력 인덱스
- **[`CLAUDE.md`](../CLAUDE.md)** (repo 루트) — 환경·구조·대회 규칙 핸드오프
