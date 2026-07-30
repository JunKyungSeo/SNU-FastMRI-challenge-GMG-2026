"""PromptMR+ config 팩토리 (SDUM의 config.py::tiny_8gb_args 대체).

이번 대회 확정 구조: cascade 10 · feature 48 · sens feature 24 · 단일 슬라이스(num_adj_slices=1) ·
Cartesian 전용. feature/prompt per-level 리스트는 n_feat0=48·sens=24에 대응하는 PromptMR+ 원본
default 표를 그대로 사용(파생 규칙 아님).

⚠️ 얕은 피라미드 유지: feature_dim은 3레벨(=다운샘플 2단)로 고정. SDUM 논문 ablation(Table 3b)에서
다운샘플 2단이 최적이고 더 깊은 피라미드는 작은 병변 대비를 약화 → 우리 점수의 50%가 SSIM_bbox(병변)
이므로 피라미드를 더 깊게 만들지 않는다.
"""
from utils.model.promptmr.promptmr_plus import PromptMR


def promptmr_kwargs(num_cascades=10, n_feat0=48, sens_n_feat0=24, n_history=3):
    """PromptMR(**kwargs)로 넘길 인자. 단일 슬라이스(num_adj_slices=1) 고정."""
    return dict(
        num_cascades=num_cascades,                     # 단일 슬라이스 전용(num_adj_slices 인자는 슬림화로 제거됨)
        n_feat0=n_feat0,
        feature_dim=[72, 96, 120],                     # feature 48 계열 (3레벨=2다운샘플, 얕은 피라미드)
        prompt_dim=[24, 48, 72],
        sens_n_feat0=sens_n_feat0,
        sens_feature_dim=[36, 48, 60],                 # sens feature 24 계열
        sens_prompt_dim=[12, 24, 36],
        n_history=n_history,                           # momentum(기여 #2). 메모리 레버(원본 11)
        n_buffer=4,                                    # adaptive gradient buffer(기여 #1)
        adaptive_input=True,                           # self-adaptive gradient(기여 #1)
    )


def build_promptmr_net(args):
    """train.py args → PromptMR 백본. CLI 레버: num_cascades / n_feat0 / sens_n_feat0 / n_history.
    ⚠️ feature_dim/prompt_dim 리스트는 **고정**이라 --n-feat0/--sens-n-feat0만 바꾸면 stem/skip/out 폭만
    바뀌고 encoder-decoder 피라미드는 48/24 기준 그대로다(비례 스케일 아님). 폭 스윕 시 리스트도 함께 조정 필요."""
    return PromptMR(**promptmr_kwargs(
        num_cascades=int(args.num_cascades),
        n_feat0=int(getattr(args, "n_feat0", 48)),
        sens_n_feat0=int(getattr(args, "sens_n_feat0", 24)),
        n_history=int(getattr(args, "n_history", 3)),
    ))
