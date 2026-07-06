def test_deps_importable():
    import monai  # noqa: F401
    import einops  # noqa: F401
    import timm  # noqa: F401
    from monai.apps.reconstruction.networks.nets.utils import divisible_pad_t  # noqa: F401


def test_sdum_imports():
    from utils.model.sdum.latent_recon import create_mri_recon_model  # noqa: F401
    from utils.model.sdum.restormer import restormer_mri, Restormer  # noqa: F401
    from utils.model.sdum.nv_varnet import CoilSensitivityModel_DCAE  # noqa: F401
