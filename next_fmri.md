1. shell scripting 지금처럼 하기 


2. final training의 augment argument 설정하는 거 default 안에 넣어서 수정하기


3. gpu 받기 전에 준비해서 돌리기
OT 끝나는 날 runpod 만들어서 혹은 연구실 GPU 로 미리 돌리기 아니면 문석이나 영민이 헤드락걸어서 빼앗기

4. 10 epoch 돌리는 run을 줄이기
 - smoke를 하거나 full run 권장

5. runpod 정산

6. pop eval 전에 full run 1번 돌리기
 - recon은 로컬에서 계속 하면서 결과 보고, 여러 smoke나 실험들은 vessl 외에서 계속 확인하기

7. full run을 돌리는 중간에 개선점 파악하고(ex. 오버피팅) 그걸 바탕으로 MoE 돌리거나, 수정하기

8. 클로드한테 repo md파일 맡기고, 우리끼리는 Notion 공유

9. 클로드 skills 깎아오기

10. merge_moe utils 안으로 옮기기

11. 무릎 y좌표 별로 짤라서 다른 moe나 서로 다른 걸로 학습시키기 (앙상블?등)