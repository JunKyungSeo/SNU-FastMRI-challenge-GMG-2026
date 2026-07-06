def test_deps_importable():
    import monai  # noqa: F401
    import einops  # noqa: F401
    import timm  # noqa: F401
    from monai.apps.reconstruction.networks.nets.utils import divisible_pad_t  # noqa: F401


def test_sdum_imports():
    from utils.model.sdum.latent_recon import create_mri_recon_model  # noqa: F401
    from utils.model.sdum.restormer import restormer_mri, Restormer  # noqa: F401
    from utils.model.sdum.nv_varnet import CoilSensitivityModel_DCAE  # noqa: F401


def test_tiny_model_param_count():
    import torch  # noqa: F401
    from utils.model.sdum.config import tiny_8gb_args, build_cascaded

    args = tiny_8gb_args(num_cascades=4, channels=[32, 64])
    model = build_cascaded(args)
    n_params = sum(p.numel() for p in model.parameters())
    # tiny config는 230M "small" 변형보다 훨씬 작아야 함
    assert n_params < 30_000_000, f"too big: {n_params/1e6:.1f}M"
    assert n_params > 100_000, f"suspiciously tiny: {n_params}"
