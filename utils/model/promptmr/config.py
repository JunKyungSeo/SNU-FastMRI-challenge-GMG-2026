# PromptMR+ tiny config (spec §3): 8GB·FP32·batch=1용 축소판. "+" 4개 전부 ON.
from typing import Sequence


def promptmr_tiny_config(num_cascades: int = 6,
                         feature_dim: Sequence[int] = (24, 32, 40),
                         n_feat0: int = 16,
                         **overrides) -> dict:
    """PromptMR(**promptmr_tiny_config(...)) 형태로 쓰는 kwargs dict를 반환.

    CLI 노출 노브 = num_cascades / feature_dim(--channels) / n_feat0.
    나머지(prompt·sens·CAB 개수, "+" 플래그)는 MVP 고정.
    """
    cfg = dict(
        num_cascades=num_cascades,
        num_adj_slices=1,                 # MVP: 단일 슬라이스(adjacent-slice는 후속)
        n_feat0=n_feat0,
        feature_dim=list(feature_dim),
        prompt_dim=[8, 16, 24],
        sens_n_feat0=8,
        sens_feature_dim=[12, 16, 20],
        sens_prompt_dim=[8, 12, 16],
        len_prompt=[5, 5, 5],
        prompt_size=[16, 8, 4],
        n_enc_cab=[2, 3, 3],
        n_dec_cab=[2, 2, 3],
        n_skip_cab=[1, 1, 1],
        n_bottleneck_cab=3,
        no_use_ca=False,
        learnable_prompt=True,            # 충실: prompt bank 학습 가능(upstream 코드 기본 False)
        adaptive_input=True,              # +
        n_buffer=4,                       # +
        n_history=3,                      # + (tiny; 논문 기본 11)
        use_sens_adj=True,                # +
    )
    cfg.update(overrides)
    return cfg
