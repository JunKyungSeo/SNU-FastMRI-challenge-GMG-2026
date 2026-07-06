from dataclasses import dataclass, field, replace


@dataclass
class SdumArgs:
    """8GB용 SDUM 설정. restormer_mri/create_mri_recon_model이 args.* 로 읽는 필드 제공.

    UC 제거(time_cond/label_cond=False), temporal 제거(num_frames=1),
    SWDC 유지(use_dc_weight_map=True), CSME per-cascade(use_single_csm=False).
    """

    # --- cascade / backbone (레버) ---
    model_type: str = "restormer"
    num_cascades: int = 4                 # A1
    channels: list = field(default_factory=lambda: [32, 64])   # A2 (2-level 기본; [C] 하나면 1-level A/B)
    num_blocks: list = field(default_factory=lambda: [3, 6])
    num_heads: list = field(default_factory=lambda: [1, 2])
    num_refinement: int = 2
    mlp_ratio: int = 3                    # GDFN 확장
    norm_type: str = "ln"
    drop_path: float = 0.1
    hybrid_attn: bool = False
    post_norm: bool = False
    ms_refinement: bool = False
    # --- temporal / coils ---
    num_frames: int = 1                   # B6: 단일 슬라이스 (knee 2D)
    num_reduced_coils: int = 1            # CSME가 1개 coil-combined 영상으로 축소
    # --- CSME (A3) ---
    use_csm: bool = True
    use_acs_region: bool = True
    use_single_csm: bool = False          # 옵션 (c)는 후속(주기적 재추정); 기본 per-cascade
    # --- SWDC (B4) 유지 ---
    use_dc_weight_map: bool = True
    mask_specific_dc_weight_map: bool = True
    # --- UC (B5) 제거 ---
    time_cond: bool = False
    label_cond: bool = False
    num_experts: object = None
    top_k: int = 1
    # --- cascade skips ---
    enable_cas_skips: bool = True
    flow: bool = False
    # --- SWDC 패밀리/프로토콜 테이블 (knee: 단일 uniform 패밀리, acc4/acc8) ---
    dataset: str = "knee"                 # -> soft_dc가 (768,768) dc map 분기 사용
    train_mask_types: list = field(default_factory=lambda: ["uniform"])
    accelerations: list = field(default_factory=lambda: [4, 8])
    acq_types: list = field(default_factory=lambda: ["knee"])
    # --- 사전학습 없음 (규정) ---
    pretrained_csm: object = None
    pretrained_recon: object = None


def tiny_8gb_args(**overrides) -> SdumArgs:
    return replace(SdumArgs(), **overrides) if overrides else SdumArgs()


def build_cascaded(args: SdumArgs):
    from utils.model.sdum.latent_recon import create_mri_recon_model
    return create_mri_recon_model(args)
