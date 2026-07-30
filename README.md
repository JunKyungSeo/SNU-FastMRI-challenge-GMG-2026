# 1. 파일 배치

repo root에 존재하는 best_model.pt 파일을 root/result/promptmr_moe/checkpoints 폴더 내에 위치시키면 됩니다. data의 경우 기존 vessel 서버와 동일한 data 구조를 가져야 합니다. 

# 1. train 실행

bash shell_scripting/shell20.sh

# 2. recon_eval.py 실행

저희가 실행하기 위해 작성한 명령어는 아래와 같습니다.

bash shell_scripting/shll20_reconstruct.sh 

혹은 기존 recon_eval.sh 파일을 수정하여 (-n "promptmr_moe")

bash recon_eval.sh

# 3. valloss_log / terminal_output

repo root에 valloss_log 및 terminal_output이라는 이름의 파일이 있습니다.
valloss_log의 경우 일반 valloss이고, terminal_output의 경우, Runpod(RTX 4000 Ada) 로 실행한 training terminal log와 vessel에서 실행한 shell20_reconstruct.sh의 recon terminal log의 합본입니다.