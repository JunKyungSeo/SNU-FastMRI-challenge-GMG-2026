"""ACS 코일 에너지 보존율 리포트 — SVD coil compression을 몇 개(Nvc)로 할지 데이터로 결정.

각 슬라이스의 ACS(중앙 완전샘플) calibration 행렬을 SVD해서, 상위 Nvc 가상코일이 담는
누적 에너지 비율(Σσ²)을 계산한다. 학습의 `coil_compress`와 **동일한 ACS 검출(_detect_acs)·
좌특이벡터 기저**를 쓰므로, 여기서 나온 보존율이 실제 압축의 정보손실과 일치한다.

bbox(작은 병변)는 꼬리 특이벡터의 국소 감도 구조에 민감하므로, 평균뿐 아니라 **worst-case
슬라이스(최소 보존율 / p5)**와 **99.5% 도달에 필요한 최소 Nvc(worst)**를 함께 본다.

사용:
    python scripts/coil_energy_report.py -p /root/Data/leaderboard --nvc 4,6,8,10,12
    python scripts/coil_energy_report.py -p /Data/val --max-vols 20 --max-slices 8
출력: training_anal/coil_energy_report.md (기본)
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np

# 학습과 동일한 ACS 검출 재사용(cv2 등 의존 로드됨 — 서버 환경 전제)
sys.path.insert(0, os.getcwd())
from utils.data.transforms import _detect_acs

ENERGY_TARGET = 0.995   # bbox-safe 기본 임계


def _kspace_dirs(data_path: Path):
    """(라벨, kspace_dir) 리스트. leaderboard(acc4/acc8) / train·val(kspace) 레이아웃 모두 처리."""
    dirs = []
    if (data_path / 'acc4').is_dir() or (data_path / 'acc8').is_dir():
        for acc in ('acc4', 'acc8'):
            kd = data_path / acc / 'kspace'
            if kd.is_dir():
                dirs.append((acc, kd))
    elif (data_path / 'kspace').is_dir():
        dirs.append((data_path.name, data_path / 'kspace'))
    else:
        dirs.append((data_path.name, data_path))
    return dirs


def _slice_retained(kspace_slice: np.ndarray, mask1d: np.ndarray, nvc_list):
    """한 슬라이스 (C,H,W) → {Nvc: 누적에너지비}, min_nvc(≥target). ACS 없으면 None."""
    C = kspace_slice.shape[0]
    acs = _detect_acs(np.asarray(mask1d))
    if int(acs.sum()) < 1:
        return None, None
    cal = kspace_slice[:, :, acs].reshape(C, -1)          # (C, H*n_acs)
    sv = np.linalg.svd(cal, compute_uv=False)             # 특이값 (내림차순)
    energy = (sv ** 2)
    total = energy.sum()
    if total <= 0:
        return None, None
    cum = np.cumsum(energy) / total                       # cum[k-1] = 상위 k개 보존율
    retained = {nvc: float(cum[min(nvc, C) - 1]) for nvc in nvc_list}
    min_nvc = int(np.searchsorted(cum, ENERGY_TARGET) + 1)  # target 도달 최소 개수
    return retained, min(min_nvc, C)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('-p', '--path-data', type=Path, required=True, help='acc4/acc8 상위 or kspace 폴더')
    ap.add_argument('--nvc', type=str, default='4,6,8,10,12', help='평가할 가상코일 수(콤마)')
    ap.add_argument('--max-vols', type=int, default=0, help='폴더당 볼륨 상한(0=전부)')
    ap.add_argument('--max-slices', type=int, default=0, help='볼륨당 슬라이스 상한(0=전부)')
    ap.add_argument('-o', '--out', type=Path, default=Path('training_anal/coil_energy_report.md'))
    args = ap.parse_args()

    import h5py
    nvc_list = sorted(int(x) for x in args.nvc.split(','))
    rows = []   # (label, C, {nvc: [보존율...]}, [min_nvc...])

    for label, kdir in _kspace_dirs(args.path_data):
        files = sorted(kdir.iterdir())
        if args.max_vols:
            files = files[:args.max_vols]
        per_nvc = {nvc: [] for nvc in nvc_list}
        min_nvcs = []
        C_seen = None
        for f in files:
            if f.suffix not in ('.h5', '.hdf5'):
                continue
            with h5py.File(f, 'r') as hf:
                ks = hf['kspace'][:]           # (S, C, H, W) complex
                mask = np.array(hf['mask'])    # (W,)
            n = ks.shape[0]
            idxs = range(n) if not args.max_slices else np.linspace(0, n - 1, min(args.max_slices, n)).astype(int)
            for s in idxs:
                retained, min_nvc = _slice_retained(ks[s], mask, nvc_list)
                if retained is None:
                    continue
                C_seen = ks.shape[1]
                for nvc in nvc_list:
                    per_nvc[nvc].append(retained[nvc])
                min_nvcs.append(min_nvc)
        if min_nvcs:
            rows.append((label, C_seen, per_nvc, min_nvcs))

    # ── 리포트 작성 ──
    lines = ["# ACS 코일 에너지 보존율 리포트", "",
             f"- 데이터: `{args.path_data}`  ·  에너지 임계(min Nvc): **{ENERGY_TARGET:.1%}**",
             f"- 각 셀 = 상위 Nvc 가상코일 누적 에너지비 **평균 (worst=p5 / 최소)**. bbox는 worst에 민감.", ""]
    header = "| set | C | " + " | ".join(f"Nvc={n}" for n in nvc_list) + " | min Nvc(≥임계) 평균/최악 |"
    lines += [header, "|" + "---|" * (len(nvc_list) + 3)]
    for label, C, per_nvc, min_nvcs in rows:
        cells = []
        for nvc in nvc_list:
            a = np.array(per_nvc[nvc])
            cells.append(f"{a.mean()*100:.2f}% ({np.percentile(a,5)*100:.2f} / {a.min()*100:.2f})")
        mn = np.array(min_nvcs)
        lines.append(f"| {label} | {C} | " + " | ".join(cells) + f" | {mn.mean():.1f} / {int(mn.max())} |")
    lines += ["",
              "## 해석",
              f"- **min Nvc(최악)** = 모든 슬라이스에서 {ENERGY_TARGET:.1%} 에너지를 담는 데 필요한 코일 수의 상한.",
              "  이 값을 **Nvc 하한**으로 삼고, 그 근처부터 recon_eval 3축(SSIM_full·SSIM_bbox·time)을 스윕해",
              "  **bbox 무회귀 + time 이득**의 elbow를 고른다.",
              "- 보존율이 Nvc에서 이미 ~100%면 그 이상은 무손실(속도만 손해), 급락 지점 아래는 bbox cliff.", ""]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines))
    print(f"[coil_energy_report] {len(rows)} set(s) → {args.out}")
    for label, C, per_nvc, min_nvcs in rows:
        mn = np.array(min_nvcs)
        print(f"  {label}: C={C}, min Nvc(≥{ENERGY_TARGET:.1%}) 평균 {mn.mean():.1f} / 최악 {int(mn.max())}")


if __name__ == '__main__':
    main()
