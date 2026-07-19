# PromptMR+ 빌더: CLI args -> PromptMR. backbone이 forward(kspace,mask)->(B,384,384)
# 계약을 이미 만족하므로 별도 forward 래퍼가 없다(k-space native, 자체 정규화).
from utils.model.promptmr.promptmr import PromptMR
from utils.model.promptmr.config import promptmr_tiny_config


def build_promptmr(args):
    cfg = promptmr_tiny_config(
        num_cascades=getattr(args, "num_cascades", 6),
        feature_dim=getattr(args, "channels", [24, 32, 40]),
        n_feat0=getattr(args, "n_feat0", 16),
    )
    return PromptMR(**cfg)
