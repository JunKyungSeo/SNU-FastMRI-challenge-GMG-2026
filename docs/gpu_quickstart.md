# GPU 서버 Quickstart — SDUM 학습 (올리자마자 시작)

master 브랜치에 SDUM 백본이 이미 들어가 있음. GPU 서버(예: VESSL 1080 8GB)에서 아래를 **위에서부터 복붙**하면 됨.
Colab에서 어댑터 forward는 **이미 검증됨** → 여기선 "이 GPU·데이터에서 도는가 + 8GB에 맞는가"만 **3분** 확인하고 바로 학습.

> ⚠️ **왜 바로 `train.py`부터 안 하나:** 8GB 적재는 실제 1080에서만 확정됨. 1~2단계(3분)를 건너뛰고 큰 config로 학습을 걸면 **iteration 1에서 CUDA OOM → 몇 시간 날림.** 3분 투자로 방지.

---

## 0. 코드 + 환경 (1회)
```bash
# 이미 repo가 있으면: cd FastMRI_SNU_2026_GMG && git checkout master && git pull
git clone -b master https://github.com/JunKyungSeo/FastMRI_SNU_2026_GMG.git
cd FastMRI_SNU_2026_GMG
pip install -r requirements.txt      # torch가 이미 있으면 monai/einops/timm만 설치됨
```

## 1. 30초 sanity — 이 환경에서 코드가 도는지
```bash
python -m pytest tests/ -v
```
→ **5개 통과면 OK.** (실패하면 학습 들어가지 말고 에러부터 해결.)

## 2. ⚠️ 8GB에 맞는 config 찾기 (2~3분, 필수)
가장 큰데 8GB에 들어가는 조합을 고른다. 각 줄이 `OK ... peak=X.XXGB` 또는 `OOM` 출력:
```bash
python scripts/probe_oom.py --cascades 4 --channels 16 32
python scripts/probe_oom.py --cascades 6 --channels 24 48
python scripts/probe_oom.py --cascades 8 --channels 32 64
```
→ **peak가 ~7.3GB 이하로 `OK`인 가장 큰 조합**을 선택. (OOM 뜨면 그건 제외.)
예시 판단: `--cascades 6 --channels 24 48`가 OK면 그걸로. 더 키우고 싶으면 `--channels 32 64`도 시도.

## 3. 🚀 학습 시작
2단계에서 고른 config로(아래는 예시 값 — 본인이 고른 걸로 교체):
```bash
python train.py \
  --model sdum \
  --num-cascades 6 --channels 24 48 \
  -b 1 -l 2e-4 -e 30 -r 50 \
  -n sdum_run1 \
  -t /root/Data/train/ \
  -v /root/Data/val/
```
- `-t`/`-v`: 서버의 knee 데이터 경로(VESSL 예: `/root/Data/train/`, `/root/Data/val/`). 기본값은 `/Data/train/`, `/Data/val/`.
- `--model sdum` **필수**(생략하면 기존 varnet으로 돌아감).
- `-b 1` 고정, `-l 2e-4`(AdamW), `-e 30`(첫 실험; 실제 제출은 더 길게).
- 체크포인트·로그: `../result/sdum_run1/`.

## 4. 첫 몇 iteration 지켜보기
- `Loss = ...`가 **유한하고 대략 감소** → 정상, 그대로 두면 됨.
- **`CUDA out of memory`** → config가 큼. 2단계로 돌아가 **더 작은 조합**(cascade↓ 또는 channels↓)으로 재시작.
- **`NaN`/`inf`** → lr 낮추기(`-l 1e-4`).
- 오래 걸리면 정상(unrolled + FP32). `-r 50`이면 50 iter마다 로그.

---

## 참고
- 8GB는 Colab T4(15GB)와 다르므로, **2단계 probe는 반드시 실제 학습할 GPU에서** 돌려야 의미 있음.
- 레버 의미·결정: `strategy/model_decision_points.md`, 크기 감: `docs/param_ranker_explained.md`.
- 이식 상세: `docs/superpowers/plans/2026-07-06-sdum-backbone-port.md`, 리스크: `CLAUDE.md`.
- 아직 안 붙은 것(다른 작업): GRAPPA prior, bbox 손실 — 지금 학습엔 없어도 됨(SSIM 손실로 학습).
