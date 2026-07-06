# Vendored SDUM model (NVIDIA NV-Raw2Insights-MRI)

출처: https://github.com/NVIDIA-Medtech/NV-Raw2insights-MRI (Apache 2.0).

- `restormer.py` / `latent_recon.py` / `nv_varnet.py` — NV `scripts/models/`에서 라이선스 헤더 포함 복사, import 경로만 우리 패키지(`utils.model.sdum.*`)로 재배선.
- `nv_utils.py` — NV `scripts/utils.py`에서 모델이 실제로 쓰는 헬퍼 6개만 AST로 추출(get_acs_region/get_acs_image/sensitivity_map_reduce/sensitivity_map_expand/reshape_channel_to_batch_dim/reshape_batch_channel_to_channel_dim). complex/fft 연산은 MONAI에서 import.

**사전학습 가중치는 사용하지 않음**(대회 규정: pretrained/external 금지). HuggingFace 자동 다운로드 코드(`CHECKPOINT_HF_REGISTRY`, `resolve_checkpoint_path`)는 의도적으로 vendor하지 않음.

로컬 환경엔 torch/MONAI 미설치라 import·pytest 실행은 실제(GPU/venv) 환경으로 미룸. 로컬에선 AST 파싱으로 문법만 검증.
