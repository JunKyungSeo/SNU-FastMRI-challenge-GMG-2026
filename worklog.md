# worklog — 팀 변경 타임라인 (누가 · 언제 · 왜)

두 팀원이 서로 다른 시간대에 이 대회 작업(전략·구현·설정)을 수정한다.
서로 뭘 왜 바꿨는지 따라잡기 위한 단일 시간순 로그다. 자리 비운 사이 뭐가 바뀌었는지 여기부터 읽는다.

작성자 태그
- @bedso = kindofwater <bedso@snu.ac.kr>
- @esel34 = seojunkyung0428 <esel343434@snu.ac.kr / sjkdanny@naver.com> (GitHub JunKyungSeo, repo 소유자)

## 쓰는 법 (규칙)

- 최신이 맨 위 (역순). 새 엔트리는 구분선 바로 아래에 추가한다.
- 이 파일의 가치는 "왜"와 "핸드오프"다. "무엇"만 적으면 git log와 중복이라 의미가 없다.
  git이 잘 기록하는 것(날짜·작성자·바뀐 파일·diff)은 최소한으로 적고, git이 못 남기는 것(의도·근거·아직 커밋 안 된 변경·상대에게 남기는 요청)에 집중한다.
- 바뀐 파일 경로는 나열하지 않는다. 그건 git이 이미 갖고 있다. 무엇을 바꿨는지는 개념으로 적고, 자세히 봐야 하면 관련 문서 한 곳으로 링크한다.
- 아직 커밋 안 된 것도 적는다. 큐 슬롯(shell_scripting/shellN.sh), 돌고 있는 실험, 전략 판단 등.
- 시각은 UTC와 KST(UTC+9)를 함께 적는다. 커밋 시각은 git log -1 --format='%cd' --date=format-local:'%Y-%m-%d %H:%M'로 뽑으면 된다.
- 마크다운 강조(별표)나 이모지는 쓰지 않는다. 이 파일은 렌더링 없이 raw 텍스트로 읽으므로 오히려 지저분해진다.
- 문장은 완결된 문장으로 쓴다. 너무 압축해서 파편처럼 만들지 않는다.
- 상세는 여기 다 쓰지 말고 링크로 넘긴다 (training_anal/...md, strategy/...md, 커밋 해시).
- 손으로 쓰기 번거로우면 Claude에게 "worklog에 오늘 내 변경 정리해줘"라고 하면 git 히스토리에서 무엇·시각·커밋을 채워준다. 사람은 왜만 확인하면 된다.

엔트리 템플릿

    ## YYYY-MM-DD HH:MM UTC / YYYY-MM-DD HH:MM KST — @태그

    한 줄 요약: 무엇을 했는지 한 문장. 이 줄만 훑어도 최근 변경이 파악되게.

    무엇: 무엇을 바꿨나. 파일 경로 나열 대신 개념으로. 자세한 건 문서 링크로.

    왜: 의도와 근거. 기각한 대안이 있으면 함께.

    상태: 커밋 <hash> / 미커밋 / 큐 등록(미실행) / 실험 진행중 등.

    핸드오프 → @상대: 상대에게 남기는 요청이나 인계. 있을 때만.

---

## 2026-07-26 14:42 UTC / 2026-07-26 23:42 KST — @bedso

한 줄 요약: kfold_split을 acc 층화 분할로 교체했다. --acc 값과 무관하게 fold 경계가 고정된다.

무엇: acc4·acc8 볼륨을 각각 k등분해 fold별로 합치는 층화 분할을 kfold_split의 기본 동작으로
넣었다. 순수 헬퍼 _stratified_folds와 pytest(scripts/tests/test_kfold_stratify.py, 서버 실행)를
같이 추가했다. shell21·shell22 헤더의 관련 경고도 현행화했다.

왜: 기존 구현은 --acc로 거른 뒤 분할해서, MoE의 pretrain(--acc all)→미세조정(--acc 4/8)을
이으면 단계마다 hold-out이 달라졌다(실측 40개 중 30개 오염, 4런 분석 문서 참고). 층화가
팀 합의된 전제 조건이었고(shell21 주석·4런 분석 §6), 이번에 기본 동작으로 교체했다.
플래그 방식(기본 off)도 검토했으나 실수로 빼먹으면 오염이 재발하므로 기본 교체를 골랐다.

주의: 교체 전 파티션으로 돈 promptmr_kfold_f1/f2와 새 런의 val은 비교 불가다. shell22의
증강 A/B는 OFF arm(f2)이 구파티션이라 baseline 재학습 없이는 무효 — 다음 run 결정에서 처리.

상태: 커밋 9f2794f(층화 헬퍼·단위 테스트) 13943cb(독립성 테스트 보강) f5b8d89(kfold_split 교체·통합 테스트)
4a2fefe(슬롯 경고·worklog). 서버 pytest·스모크 검증은 아직(아래 핸드오프).

핸드오프 → @esel34: 서버에서 python -m pytest scripts/tests/test_kfold_stratify.py -v 와
SMOKE=1 FOLD=0 bash shell_scripting/shell21.sh 를 돌려달라. 확인할 것은 [K-Fold] 로그의 val
구성이 fold를 바꿔도 acc4·acc8 개수가 똑같이 나오는지, 그리고 그 값이 같은 줄에 찍히는
train+val 총계를 k로 나눈 값과 맞는지다. 층화의 서명이 바로 그 두 가지다. 구체적 숫자를
미리 적지 않는 이유는 실제 acc4/acc8 볼륨 수에 따라 달라져서다. 예전 f2의 22 대 18 같은
값을 기준으로 삼으면 정상 결과를 실패로 읽게 된다.

## 2026-07-19 01:42 UTC / 2026-07-19 10:42 KST — @esel34

한 줄 요약: 증강 RNG의 프로세스 간 재현성 버그를 찾아 고쳤다(현재는 잠복 상태).

무엇: MRIAugmentor의 샘플별 RNG가 builtin hash((seed, epoch, str(fname), slice))를 썼는데, Python3의 문자열 해시 salt(PYTHONHASHSEED 미설정) 때문에 프로세스마다 시드가 달라졌다. zlib.crc32(repr(...))로 교체해 재실행 간 동일 추출을 확인했다. 관련 기록 문서(gaps_and_blindspots, summary 6절, augmentation_and_preprocessing)도 갱신했다.

왜: "샘플별 결정적 RNG" 주석 약속과 대회 재현성 규칙(seed 고정 시 재학습 같은 결과)이 증강 ON일 때 깨졌다. seed_fix로도 안 잡히는 구멍이라 별도 수정이 필요했다.

상태: 커밋 6acad53. 증강은 아직 dead(aug_on 기본 False, set_epoch 미호출)라 현재 런타임 동작 변화는 없고, 배선하는 순간부터 재현성이 보장된다.

## 2026-07-18 16:00 UTC / 2026-07-19 01:00 KST — @esel34

한 줄 요약: 백본을 SDUM에서 PromptMR+로 교체하고 손실·옵티마이저·coil compression을 새로 배선했다. 이어서 문서와 파일 구조도 정리했다.

무엇: 작년 대회 우승 코드(sMartify-fastmri/2025_final)의 PromptMR+를 utils/model/promptmr/로 벤더링하고, knee 단일 슬라이스 Cartesian 전용으로 슬림화했다(인접슬라이스 SME와 비-Cartesian 분기 제거, 원본과 수치 동일 검증). 손실은 masked SSIM에 foreground 면적비의 제곱근을 곱하는 방식(--area-weight), 옵티마이저는 AdamW 1e-6에 gradient clipping과 accumulation을 더했다. SVD coil compression을 학습·추론 양쪽에 배선하고, 압축 개수를 정하기 위한 ACS 에너지 리포트와 recon_eval 스코어카드 스크립트를 추가했다. SDUM 코드는 삭제하지 않고 보존하되 --model 배선에서만 뺐다. 배선 상세와 실행법은 strategy/promptmr_plus.md에 있다.

왜: SDUM 계열 성능이 정체됐고, PromptMR+의 self-adaptive gradient와 momentum이 정적 knee에도 유효하다(논문 검증). coil compression은 8GB에서 cascade10 feature48 모델의 VRAM과 추론시간을 줄이는 레버인데, 너무 많이 압축하면 bbox(작은 병변)가 먼저 무너지므로 ACS 에너지 보존율로 하한을 잡고 3축으로 스윕해 정한다.

상태: 커밋 19093fe(백본 교체) 8fd4e40(README). coil compression 배선과 파일 정리(HANDOFF·SDUM 전용 테스트 삭제, scripts/tests 분리, docs/archive 신설)는 아직 미커밋. GPU 실검증(OOM 확인, recon_eval 3축)은 아직 안 함 — 이 세션은 로컬 맥이라 CPU 스모크까지만 했다.

핸드오프 → @bedso: 서버에서 먼저 미니 학습으로 OOM부터 확인해야 한다. cascade10 feature48은 8GB에 안 들어갈 수 있어, 안 되면 n_history를 줄이거나(3에서 1) readout-crop을 켜거나 coil compression을 쓴다. 실행 커맨드는 strategy/promptmr_plus.md와 README 5절.

## 2026-07-11 18:13 UTC / 2026-07-12 03:13 KST — @esel34

한 줄 요약: B5 효과를 확인할 A/B 실험을 세팅했다. off-팔은 지금 도는 run10을 재활용하고, on-팔(run11)만 새로 짧게 돌린다.

B5란: strategy/preprocessing.md의 후보 번호로, 손실과 검증을 대회 채점과 똑같이 foreground(몸통) 영역 안에서만 계산하는 것을 말한다. 코드에서는 train.py의 --fg-loss 플래그이고, 기본은 꺼져 있다. 채점은 배경을 마스크로 날린 뒤 SSIM_full을 재는데 기존 손실·검증은 배경까지 포함한 전체 384²로 재고 있어서, 배경이 점수를 부풀리고 검증값이 실제 채점을 예측하지 못했다. 구현 경위는 바로 아래 07-11 16:21 엔트리, 근거와 남은 검증 항목은 preprocessing.md의 B5 항목을 보면 된다. 참고로 model_decision_points.md에도 B5가 나오는데 그건 UC(universal conditioning) 제거를 가리키는 별개 항목이라 번호만 같다.

무엇: run10(B5 꺼진 학습)의 2에폭 시점 체크포인트를 따로 복사해 off-팔로 확보했고, 같은 설정에 B5만 켠 run11을 새 큐 슬롯(shell10.sh)에 준비했다. run11은 2에폭까지 돌면 스스로 멈추도록(self-stop) 만들어, off-팔과 같은 지점에서 비교되게 했다. 겸사겸사 4일째 정지돼 있던 옛 프로세스(run4)도 정리했다.

왜: B5를 풀 학습으로 off/on 두 번 돌리면 며칠이 걸린다. 지금 도는 run10이 이미 off-팔이라, on-팔만 2에폭 돌리면 훨씬 빨리 방향을 볼 수 있다. 2에폭에서 멈추는 건 off-팔 체크포인트가 그 시점이라 맞춘 것이고, 초반 LR 곡선을 run10과 똑같이 두려고 스케줄(-e 10)은 유지한 채 학습만 멈춘다.

상태: shell10.sh 커밋. run11은 아직 실행 안 함. 큐가 자동으로 이어 물리지 않으니(README상 수동 체인) run10 끝난 뒤 직접 bash shell_scripting/shell10.sh를 돌려야 한다. off-팔 체크포인트는 result 폴더(git 밖)에 복사돼 있다.

핸드오프 → @bedso: run10 끝나면 shell10.sh를 돌리면 run11이 2에폭 후 자동 종료된다. 그다음 recon_eval로 off(sdum_run10_off2ep)와 on(sdum_run11)을 채점해서 SSIM_full이 오르는지, bbox가 안 떨어지는지 비교하면 된다.

## 2026-07-11 16:21 UTC / 2026-07-12 01:21 KST — @esel34

한 줄 요약: foreground 영역 안에서만 손실·검증을 계산하는 B5를 --fg-loss 옵션으로 구현했다 (기본 꺼짐).

무엇: 손실 함수와 검증을 대회 채점과 같은 방식으로 맞췄다. 채점은 몸통(foreground) 안에서만 SSIM을 평균내는데, 손실과 검증도 같은 영역만 보도록 바꿨다. 마스크는 채점 코드에 이미 있는 함수를 그대로 가져다 쓴다.

왜: 기존에는 손실과 검증이 배경까지 포함한 전체 이미지로 SSIM을 계산해서 실제 채점 방식과 어긋나 있었다. 배경이 점수를 부풀리고, 검증 숫자가 리더보드 점수를 제대로 예측하지 못하는 문제가 있었다.

상태: 아직 커밋 안 함. 검증은 끝났다. 마스크 손실이 채점식과 소수점 여덟 자리까지 일치하고, 옵션을 끄면 기존 학습과 완전히 동일하게 돈다. 관련 전략 문서도 함께 갱신했다. 자세한 배선은 implementation.md의 3c 항목 참고.

핸드오프 → @bedso: 실제 점수가 오르는지는 아직 모른다. run10이 끝나면 옵션을 켠 학습과 끈 학습을 같은 조건으로 돌려 비교해야 한다. 이때 bbox 점수가 떨어지지 않는지 꼭 확인.

## 2026-07-11 04:47 UTC / 2026-07-11 13:47 KST — @bedso

한 줄 요약: run9(epoch 2→4) 분석을 문서로 정리하고 run8을 독립 문서로 분리했다.

무엇: run9 학습 결과를 분석해 정리했다. 상세 분석은 training_anal의 run9 문서에 있다.

왜: 더 오래 학습했더니 검증 점수가 조금 더 올랐다(+0.0044). "작은 모델을 오래 학습하는 게 큰 모델을 짧게 하는 것보다 낫다"는 근거를 정리해 다음 run 방향을 잡으려는 것이다.

상태: 커밋 6bf6679.

핸드오프 → @esel34: run10을 준비해 뒀다 (shell9, run9와 같은 설정에 epoch만 10, 커밋 9025bd2). 끝나면 recon_eval로 세 축(full SSIM, bbox, 시간) 채점 부탁한다. full SSIM 우선.

## 2026-07-13 17:55 KST - @bedso

한 줄 요약 : run10 종료 run10 eval, leaderboard 등록, run11 시작

무엇 : run 10 종료 run 11 수정(2epoch 종료 -> 10epoch 시작) run 10 eval 돌리고 있고 run11 이후에 시작할것임

왜 : 그냥

상태 : 커밋 안됨

## 관련 문서
- strategy/collaboration_workflow.md — 작업 절차(train→분석→코멘트→반영 루프). worklog는 그 흐름의 시간순 인덱스.
- training_anal/ — 실험별 상세 분석 (worklog에서 링크로 참조).
- strategy/README.md · strategy/confirmed_model.md — 전략 인덱스·확정 설계.
