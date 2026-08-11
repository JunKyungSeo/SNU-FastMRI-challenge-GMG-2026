"""인접 슬라이스 상관 재측정 — model_strategy.md §3의 근거를 재현 가능하게 만든다.

기존 근거의 문제: 측정 스크립트가 scratchpad 임시 파일이라 사라졌고, 표본이 "제공 파일
1개씩"(acc4 1개 + acc8 1개)이었다. 그 위에서 PromptMR 간판 기능(2.5D 인접입력)을 기각했다.

⚠️ 이 스크립트가 추가로 재는 것 = **대조군(control)**. 원 측정의 결론
   "bbox 상관 0.08~0.59 → 병변이 슬라이스마다 변해 뭉개진다"는 **통계적 함정일 수 있다**:
   Pearson은 패치 자체의 분산으로 정규화하므로, 작고 대비 낮은 패치는 두 슬라이스가
   거의 같아도 상관이 낮게 나온다. 그래서 병변 bbox와 **같은 크기의 무작위 조직 박스**를
   같은 슬라이스 쌍에서 함께 재서 비교한다.
     · bbox ≈ control  → 낮은 상관은 '작은 패치' 탓 = 병변 근거 아님 → 기각 근거 무효
     · bbox <  control  → 병변이 실제로 더 빨리 변한다 = 기각 근거 유효
   SSIM(채점축과 같은 지표)도 같이 낸다 — 순위가 SSIM으로 매겨지므로 이게 더 직접적이다.

실행(데이터 있는 서버에서, GPU 불필요):
    python scripts/adjacent_slice_corr.py -p /root/Data/train
    python scripts/adjacent_slice_corr.py -p /root/Data/leaderboard --gaps 1 2 3
자기검증:
    python scripts/adjacent_slice_corr.py --selftest
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")   # Windows cp949 콘솔에서 →·← 출력 깨짐 방지

# repo 루트를 import 경로에 (scripts/ 에서 직접 실행해도 utils 를 찾게 — bbox_ssim_breakdown.py 관례)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def pearson(a, b):
    """두 패치의 Pearson r. 어느 쪽이든 분산 0이면 정의 불가 → nan."""
    a = a.ravel().astype(np.float64)
    b = b.ravel().astype(np.float64)
    a -= a.mean()
    b -= b.mean()
    da, db = np.sqrt((a * a).sum()), np.sqrt((b * b).sum())
    if da < 1e-12 or db < 1e-12:
        return np.nan
    return float((a * b).sum() / (da * db))


def ssim_patch(a, b, data_range):
    """패치 SSIM(전역 통계 1개, win 없음) — 채점 SSIM과 정의가 다르므로 참고용.
    작은 박스에 7x7 윈도가 안 들어가는 경우가 많아 채점식을 그대로 못 쓴다."""
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    ma, mb = a.mean(), b.mean()
    va, vb = a.var(), b.var()
    cov = ((a - ma) * (b - mb)).mean()
    return float(((2 * ma * mb + c1) * (2 * cov + c2))
                 / ((ma ** 2 + mb ** 2 + c1) * (va + vb + c2)))


def random_boxes(rng, fg, h, w, n):
    """foreground 안에 완전히 들어가는 (h,w) 박스를 n개 뽑는다. 병변 bbox의 대조군."""
    H, W = fg.shape
    out = []
    for _ in range(n * 40):
        if len(out) >= n:
            break
        y = int(rng.integers(0, max(1, H - h)))
        x = int(rng.integers(0, max(1, W - w)))
        if fg[y:y + h, x:x + w].mean() > 0.95:   # 배경 섞이면 상관이 인위적으로 오른다
            out.append((y, x))
    return out


def measure(root, gaps, n_control, seed, limit):
    import h5py
    from utils.common.metrics import foreground_mask

    rng = np.random.default_rng(seed)
    rows = []           # 병변 bbox
    ctrl = []           # 대조군 박스
    glob = []           # 전역(foreground 마스크 적용 / 미적용)
    files = sorted(Path(root).rglob("*.h5"))
    files = [f for f in files if "image" in str(f)]
    if limit:
        files = files[:limit]
    if not files:
        raise SystemExit(f"image h5를 못 찾았다: {root}")

    for fp in files:
        acc = "acc8" if "acc8" in fp.name else "acc4"
        with h5py.File(fp, "r") as hf:
            if "image_label" not in hf:
                continue
            vol = hf["image_label"]
            n_sl = vol.shape[0]
            dr = float(hf.attrs.get("max", vol[0].max()))
            ann = json.loads(hf.attrs["annotations"]) if "annotations" in hf.attrs else {}

            for gap in gaps:
                for s in range(n_sl - gap):
                    boxes = ann.get(str(s), [])
                    if not boxes:
                        continue                      # 병변 있는 슬라이스만 (원 측정과 동일)
                    a = np.asarray(vol[s], dtype=np.float32)
                    b = np.asarray(vol[s + gap], dtype=np.float32)
                    fg = foreground_mask(a) > 0

                    glob.append(dict(acc=acc, gap=gap,
                                     r_all=pearson(a, b),
                                     r_fg=pearson(a[fg], b[fg])))

                    for bx in boxes:
                        y, x = int(bx["y"]), int(bx["x"])
                        h, w = int(bx["height"]), int(bx["width"])
                        pa, pb = a[y:y + h, x:x + w], b[y:y + h, x:x + w]
                        if pa.size == 0:
                            continue
                        rows.append(dict(acc=acc, gap=gap, label=bx.get("label", "?"),
                                         area=h * w,
                                         r=pearson(pa, pb),
                                         ssim=ssim_patch(pa, pb, dr)))
                        for (cy, cx) in random_boxes(rng, fg, h, w, n_control):
                            qa, qb = a[cy:cy + h, cx:cx + w], b[cy:cy + h, cx:cx + w]
                            ctrl.append(dict(acc=acc, gap=gap, area=h * w,
                                             r=pearson(qa, qb),
                                             ssim=ssim_patch(qa, qb, dr)))
    return files, glob, rows, ctrl


def _stat(vals):
    v = np.asarray([x for x in vals if np.isfinite(x)], dtype=np.float64)
    if v.size == 0:
        return "  n=0"
    return (f"n={v.size:4d}  중앙 {np.median(v):.3f}  평균 {v.mean():.3f}"
            f"  [{v.min():.3f}, {v.max():.3f}]")


def report(files, glob, rows, ctrl, gaps):
    print(f"\n파일 {len(files)}개 · 병변 박스 표본 {len(rows)}개 · 대조군 {len(ctrl)}개")
    print("(원 측정은 파일 2개였다 — model_strategy.md §3)")
    for acc in ("acc4", "acc8"):
        for gap in gaps:
            g = [d for d in glob if d["acc"] == acc and d["gap"] == gap]
            b = [d["r"] for d in rows if d["acc"] == acc and d["gap"] == gap]
            c = [d["r"] for d in ctrl if d["acc"] == acc and d["gap"] == gap]
            bs = [d["ssim"] for d in rows if d["acc"] == acc and d["gap"] == gap]
            cs = [d["ssim"] for d in ctrl if d["acc"] == acc and d["gap"] == gap]
            if not g:
                continue
            print(f"\n=== {acc}  gap{gap} ===")
            print(f"  전역 r (배경포함) : {_stat([d['r_all'] for d in g])}")
            print(f"  전역 r (fg만)     : {_stat([d['r_fg'] for d in g])}")
            print(f"  병변 bbox r       : {_stat(b)}")
            print(f"  대조군 박스 r     : {_stat(c)}   ← bbox와 비교할 기준")
            print(f"  병변 bbox SSIM    : {_stat(bs)}")
            print(f"  대조군 박스 SSIM  : {_stat(cs)}")
            mb = np.median([x for x in b if np.isfinite(x)]) if b else np.nan
            mc = np.median([x for x in c if np.isfinite(x)]) if c else np.nan
            if np.isfinite(mb) and np.isfinite(mc):
                verdict = ("병변이 대조군보다 유의하게 낮다 → 기각 근거 유효"
                           if mb < mc - 0.05 else
                           "병변 ≈ 대조군 → 낮은 상관은 '작은 패치' 탓, 병변 근거 아님")
                print(f"  판정: bbox {mb:.3f} vs control {mc:.3f}  →  {verdict}")


def selftest():
    """대조군 논리가 성립하는지: 같은 이미지를 살짝 흔들면 큰 패치보다 작은 패치의
    Pearson이 더 낮게 나온다(분산 정규화 효과) — 이게 원 측정의 함정 가설이다."""
    rng = np.random.default_rng(0)
    img = rng.random((384, 384)).astype(np.float32)
    smooth = np.cumsum(np.cumsum(img, 0), 1) / (384 * 384)      # 저주파(대비 낮은) 성분
    noisy = smooth + rng.normal(0, smooth.std() * 0.1, smooth.shape)
    big = pearson(smooth, noisy)
    small = pearson(smooth[100:130, 100:150], noisy[100:130, 100:150])
    assert big > small, f"큰 패치가 더 높아야 한다: big={big:.3f} small={small:.3f}"
    assert np.isnan(pearson(np.ones((4, 4)), np.ones((4, 4)))), "분산 0이면 nan이어야 한다"
    assert abs(pearson(img, img) - 1.0) < 1e-9
    assert abs(ssim_patch(img, img, 1.0) - 1.0) < 1e-9
    print(f"selftest OK — 전역 r={big:.3f} vs 작은패치 r={small:.3f} "
          f"(동일 신호인데 패치가 작을수록 낮다 = 함정 가설 성립)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-p", "--path", help="데이터 루트 (예: /root/Data/train)")
    ap.add_argument("--gaps", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--n-control", type=int, default=5, help="병변 박스당 대조군 박스 수")
    ap.add_argument("--seed", type=int, default=430)
    ap.add_argument("--limit", type=int, default=0, help="파일 수 제한(0=전부)")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
    elif a.path:
        report(*measure(a.path, a.gaps, a.n_control, a.seed, a.limit), gaps=a.gaps)
    else:
        ap.error("-p 또는 --selftest 중 하나가 필요하다")
