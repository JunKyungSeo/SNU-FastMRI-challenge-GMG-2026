# shell_scripting/ — GPU 작업 큐

GPU를 쓰는 명령은 **여기 스크립트에서만** 실행한다.
가벼운 python(예: `h5_viewer.py`)은 큐에 넣지 말고 옆 분할 터미널에서 직접 실행.

## 큐 사용법

슬롯을 순서대로 채우고 세미콜론으로 이어서 실행:

```bash
for i in $(seq 1 10); do bash
  shell_scripting/shell$i.sh; done
```

다음에 다른 걸 돌리고 싶으면 **비어 있는 다음 슬롯**(`shell4.sh` …)에 명령을 미리 써두면 된다.
`;` 로 잇기 때문에 앞 슬롯이 실패해도 다음 슬롯은 계속 실행된다(멈추고 싶으면 `&&` 로 이을 것).

## 각 슬롯 공통 규칙

1. 맨 위 주석에 **목적** 한 줄.
2. 실행하면 `[START] <슬롯>` 배너로 **어디서 시작됐는지** echo.
3. stdout+stderr 가 `logs/<슬롯>[_f<FOLD>]_<YYYYMMDD_HHMMSS>.log` 로 남는다 (터미널에도 동시 출력). shell19/21/22처럼 K-Fold를 쓰는 슬롯은 `logs/${SLOT}_f${FOLD}_${TS}.log` 형태로 fold가 파일명에 들어간다.
4. 끝나면 `[DONE ] <슬롯> ... exit=<코드>` 로 종료코드 고지.
5. 실행되는 명령이 `set -x`로 로그에 그대로 찍힌다 — 어떤 config·`--num-workers`로 돌렸는지 **로그만 봐도** 안다. (슬롯이 `_template.sh` 기준으로 만들어졌을 때. 기존 슬롯은 `cp _template.sh shellN.sh`로 갱신.)
6. **GPU 학습엔 식별 가능한 이름 필수** — 현행 관례는 `-n promptmr_kfold_f<FOLD>[_변주] --run-tag pmr_kf_f<FOLD>[_변주]`(예: `pmr_kf_f2` ↔ 증강 ON arm `pmr_kf_f2_aug`), 스모크는 `_smoke` suffix를 붙인다. 다음 번호가 필요한 옛 관례(`runN`)를 쓸 경우엔 `training_anal/CLAUDE.md` run 이력 표에서 "마지막+1".
7. **`-r 100` 고정**(≈7분마다 heartbeat). 단 `--max-iters N` 스모크는 예외 — N보다 작은 `-r`(5~10)이라야 capped iter가 로그에 찍힌다.

## 파일

| 파일 | 역할 |
|---|---|
| `shell1.sh` ~ `shell10.sh` | 큐 슬롯. 비어 있으면 실행 시 `[TODO]` 로 티나게 실패한다. |
| `_template.sh` | 슬롯 11개째부터 복사해서 쓰는 원본. |
| `logs/` | 실행 로그 (git 미추적). |

새 슬롯이 더 필요하면:

```bash
cp shell_scripting/_template.sh shell_scripting/shell11.sh
```

## 로그 보기

```bash
tail -f shell_scripting/logs/shell1_*.log     # 실행 중 실시간
ls -t  shell_scripting/logs/                  # 최신 로그부터
```
