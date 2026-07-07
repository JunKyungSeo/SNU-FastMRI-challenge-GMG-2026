# shell_scripting/ — GPU 작업 큐

GPU를 쓰는 명령은 **여기 스크립트에서만** 실행한다.
가벼운 python(예: `h5_viewer.py`)은 큐에 넣지 말고 옆 분할 터미널에서 직접 실행.

## 큐 사용법

슬롯을 순서대로 채우고 세미콜론으로 이어서 실행:

```bash
bash shell1.sh ; bash shell2.sh ; bash shell3.sh
```

다음에 다른 걸 돌리고 싶으면 **비어 있는 다음 슬롯**(`shell4.sh` …)에 명령을 미리 써두면 된다.
`;` 로 잇기 때문에 앞 슬롯이 실패해도 다음 슬롯은 계속 실행된다(멈추고 싶으면 `&&` 로 이을 것).

## 각 슬롯 공통 규칙

1. 맨 위 주석에 **목적** 한 줄.
2. 실행하면 `[START] <슬롯>` 배너로 **어디서 시작됐는지** echo.
3. stdout+stderr 가 `logs/<슬롯>_<YYYYMMDD_HHMMSS>.log` 로 남는다 (터미널에도 동시 출력).
4. 끝나면 `[DONE ] <슬롯> ... exit=<코드>` 로 종료코드 고지.

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
