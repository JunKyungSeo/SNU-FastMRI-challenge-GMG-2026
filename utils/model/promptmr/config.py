"""PromptMR+ config 팩토리 (SDUM의 config.py::tiny_8gb_args 대체).

이번 대회 확정 구조: cascade 10 · feature 48 · sens feature 24 · 슬라이스 수는 CLI(--num-adj-slices, 기본 1) ·
Cartesian 전용. feature/prompt per-level 리스트는 base 폭에서 비율로 파생한다 — n_feat0=48·sens=24를
넣으면 PromptMR+ 원본 default 표와 **정확히 같은 값**이 나온다(_pyramid 참고).

⚠️ 얕은 피라미드 유지: feature_dim은 3레벨(=다운샘플 2단)로 고정. SDUM 논문 ablation(Table 3b)에서
다운샘플 2단이 최적이고 더 깊은 피라미드는 작은 병변 대비를 약화 → 우리 점수의 50%가 SSIM_bbox(병변)
이므로 피라미드를 더 깊게 만들지 않는다. 파생 규칙은 폭만 넓히고 **깊이는 건드리지 않는다.**
"""
from utils.model.promptmr.promptmr_plus import PromptMR


def _pyramid(base):
    """base 폭 → (feature_dim, prompt_dim).

    비율은 원본 default 표에서 그대로 읽어낸 것이다 — feature = base×[1.5, 2, 2.5],
    prompt = base×[0.5, 1, 1.5]. base=48 → [72,96,120]/[24,48,72], base=24 → [36,48,60]/[12,24,36]
    으로 원본과 일치한다. 즉 이 파생은 새 규약이 아니라 기존 표를 식으로 옮긴 것이다.

    4의 배수를 요구하는 건 0.5·1.5 배가 정수로 떨어지고 모든 레벨 폭이 짝수로 남게 하기 위해서다
    (CAB의 reduction=2 나눗셈). 지금까지 쓰인 값 16·24·48은 전부 만족한다."""
    if base % 4:
        raise ValueError(f"n_feat0/sens_n_feat0 는 4의 배수라야 한다(현재 {base}).")
    return [base * 3 // 2, base * 2, base * 5 // 2], [base // 2, base, base * 3 // 2]


def promptmr_kwargs(num_cascades=10, n_feat0=48, sens_n_feat0=24, n_history=3, num_adj_slices=1):
    """PromptMR(**kwargs)로 넘길 인자. num_adj_slices=1(기본)이면 단일 슬라이스."""
    feature_dim, prompt_dim = _pyramid(n_feat0)
    sens_feature_dim, sens_prompt_dim = _pyramid(sens_n_feat0)
    return dict(
        num_cascades=num_cascades,
        num_adj_slices=num_adj_slices,                 # 1=단일, 3=±1 인접(2.5D). model_strategy §3.1
        n_feat0=n_feat0,
        feature_dim=feature_dim,                       # n_feat0=48 → [72, 96, 120] (원본 표와 동일)
        prompt_dim=prompt_dim,                         # n_feat0=48 → [24, 48, 72]
        sens_n_feat0=sens_n_feat0,
        sens_feature_dim=sens_feature_dim,             # sens=24 → [36, 48, 60]
        sens_prompt_dim=sens_prompt_dim,               # sens=24 → [12, 24, 36]
        n_history=n_history,                           # momentum(기여 #2). 메모리 레버(원본 11)
        n_buffer=4,                                    # adaptive gradient buffer(기여 #1)
        adaptive_input=True,                           # self-adaptive gradient(기여 #1)
    )


def build_promptmr_net(args):
    """train.py args → PromptMR 백본. CLI 레버: num_cascades / n_feat0 / sens_n_feat0 / n_history.
    --n-feat0/--sens-n-feat0 를 올리면 encoder-decoder 피라미드 폭도 같은 비율로 함께 따라온다(_pyramid).
    48/24 에서는 기존과 완전히 같은 값이라 그 값으로 학습한 체크포인트는 그대로 로드된다.
    ⚠️ 48 이외(옛 hspromptmr 계열의 n_feat0=16 등)로 학습한 체크포인트는 리스트가 고정이던 시절과
    폭이 달라져 load_state_dict 에서 shape 불일치로 죽는다(조용히 틀리지는 않는다)."""
    return PromptMR(**promptmr_kwargs(
        num_cascades=int(args.num_cascades),
        n_feat0=int(getattr(args, "n_feat0", 48)),
        sens_n_feat0=int(getattr(args, "sens_n_feat0", 24)),
        n_history=int(getattr(args, "n_history", 3)),
        num_adj_slices=int(getattr(args, "num_adj_slices", 1)),
    ))
