# shell_scripting/ — GPU 작업 큐

GPU를 쓰는 명령은 **여기 스크립트에서만** 실행한다.
가벼운 python(예: `h5_viewer.py`)은 큐에 넣지 말고 옆 분할 터미널에서 직접 실행.

## 큐 사용법

사용자가 직접 CLI 에
```bash
bash shell_scripting/shell11.sh ; bash shell_scripting/_notify.sh "완료" 
```
같은 식으로 입력함.

## 각 슬롯 공통 규칙

1. 맨 위 주석에 **목적** ***한 줄*** 너무 길게 쓰지 마.
2. 새 슬롯은 `_train_template.sh`(학습) / `_evaluation_template.sh`(채점) 복사해서 시작. 고정 인자는 템플릿의 `[고정 인자]` 블록에 있고, `[이번 run]` 블록과 실행부 델타만 채운다.
3. 실행하면 `[START] <슬롯>` 배너로 **어디서 시작됐는지** echo.
4. stdout+stderr 가 `logs/<슬롯>[_f<FOLD>]_<YYYYMMDD_HHMMSS>.log` 로 남는다 (터미널에도 동시 출력). shell19/21/22처럼 K-Fold를 쓰는 슬롯은 `logs/${SLOT}_f${FOLD}_${TS}.log` 형태로 fold가 파일명에 들어간다.
5. 끝나면 `[DONE ] <슬롯> ... exit=<코드>` 로 종료코드 고지.
6. **모든 학습엔 식별 가능한 이름 필수** — 현행 관례는 다양함(정해진 규칙 없음)`-n promptmr_kfold_f<FOLD>[_변주] --run-tag pmr_kf_f<FOLD>[_변주]`(예: `pmr_kf_f2` ↔ 증강 ON arm `pmr_kf_f2_aug`), 스모크는 `_smoke` suffix를 붙인다. 다음 번호가 필요한 옛 관례(`runN`)를 쓸 경우엔 `training_anal/CLAUDE.md` run 이력 표에서 "마지막+1".

## 파일

| 파일 | 역할 |
|---|---|
| `shellN.sh` | 큐 슬롯. 종료된 슬롯은 `legacy/`로 옮긴다(역할 표: `legacy/README.md`). |
| `_train_template.sh` | 학습 슬롯 원본. 결과 덮어쓰기 가드만 있음(이어서 `--resume`, 덮어쓰기 `FORCE=1`). |
| `_evaluation_template.sh` | reconstruct+채점 슬롯 원본. 점수 사본을 `results/`에 남긴다. |
| `results/` | recon_eval 점수 사본 (작은 txt). **커밋해서 전 환경 공유** — 이게 점수의 정본. |
| `notify.sh` | 작업 끝났을 때 Gmail 알림 발송 (큐 슬롯 아님). 최초 1회 파일 상단 [설정] 채우기. `bash shellN.sh; bash notify.sh "제목"` 처럼 뒤에 이어 쓴다. |
| `logs/` | 실행 로그 (**git 추적함** — `.gitignore`에서 해제). |
| `legacy/` | 종료된 슬롯 보관. |

새 슬롯이 더 필요하면:

```bash
cp shell_scripting/_train_template.sh shell_scripting/shell24.sh        # 학습
cp shell_scripting/_evaluation_template.sh shell_scripting/shell24.sh   # 채점
```

## 로그 보기

```bash
tail -f shell_scripting/logs/shell21_*.log    # 실행 중 실시간
ls -t  shell_scripting/logs/                  # 최신 로그부터
```
