"""SDUM 근사 파라미터/메모리 랭커 (torch 불필요, 근사값=상대 비교용).

절대 MB가 아니라 "어느 config가 더 큰가 / 8GB 위험한가"를 가르는 용도.
정밀값은 estimate_sdum.py(venv) 또는 GPU probe_oom.py에서.
"""
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")   # Windows cp949 콘솔에서 한글/em-dash 출력
except Exception:
    pass


def restormer_level_params(c, blocks, mlp):
    # TransformerBlock 근사: MDTA(qkv 1x1: c*3c, dwconv 3x3: 3c*9, proj: c*c)
    mdta = c * (3 * c) + (3 * c) * 9 + c * c
    # GDFN(project_in 1x1: c*2h, dwconv 3x3: 2h*9, project_out 1x1: h*c), h=c*mlp
    h = c * mlp
    gdfn = c * (2 * h) + (2 * h) * 9 + h * c
    return blocks * (mdta + gdfn)


def restormer_params(channels, num_blocks, num_refinement, mlp):
    L = len(channels)
    p = 0
    p += 2 * channels[0] * 9          # embed_conv 3x3 (in=2)
    for i in range(L):
        p += restormer_level_params(channels[i], num_blocks[i], mlp)   # encoders
    for i in range(L - 1):
        p += channels[i] * (channels[i] // 2) * 9          # down conv 3x3
        p += channels[i + 1] * (channels[i + 1] * 2) * 9   # up conv 3x3
        p += 2 * channels[i] * channels[i]                 # reduce 1x1
        p += restormer_level_params(channels[i], num_blocks[i], mlp)   # decoders
    p += restormer_level_params(channels[0], num_refinement, mlp)      # refinement
    p += channels[0] * 2 * 9          # output 3x3
    return p


CSME_PARAMS = 1_500_000   # ComplexUnet_DCAE {12,24,48,96,192} 고정, 대략치


def total_params(cascades, channels, num_blocks, num_refinement, mlp):
    per = restormer_params(channels, num_blocks, num_refinement, mlp) + CSME_PARAMS
    return cascades * per


def activation_score(cascades, channels, H=384, W=372):
    # 상대 점수: Σ(채널 × 공간) × cascade (checkpointing으로 실제론 완화됨)
    L = len(channels)
    s = 0
    for i in range(L):
        s += channels[i] * (H // (2 ** i)) * (W // (2 ** i))
    return cascades * s


if __name__ == "__main__":
    num_blocks = [3, 6]
    num_refinement = 2
    mlp = 3
    cascade_grid = [2, 4, 6, 8]
    channel_grid = [[16, 32], [24, 48], [32, 64]]
    print("== 근사 파라미터 수 (M) — 절대치 아님, 상대 비교용 ==")
    print(f"{'T':>3} | " + " | ".join(f"{str(c):>9}" for c in channel_grid))
    for T in cascade_grid:
        row = [f"{total_params(T, ch, num_blocks, num_refinement, mlp)/1e6:>7.1f}M" for ch in channel_grid]
        print(f"{T:>3} | " + " | ".join(f"{r:>9}" for r in row))
    print("\n== 상대 활성값-메모리 점수 (클수록 VRAM 위험) ==")
    for T in cascade_grid:
        row = [f"{activation_score(T, ch)/1e6:>7.0f}" for ch in channel_grid]
        print(f"{T:>3} | " + " | ".join(f"{r:>9}" for r in row))
