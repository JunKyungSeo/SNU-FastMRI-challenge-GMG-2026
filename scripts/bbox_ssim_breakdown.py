"""Stage-0 diagnosis B: re-aggregate SSIM from the SAVED run10 reconstructions.

CPU only. No model, no reconstruction. Reads
  /root/result/sdum_run10/reconstructions_leaderboard/acc{4,8}/*.h5  ['reconstruction']
and the leaderboard targets, then calls the FIXED utils/common/metrics.py
(ssim_full / ssim_bbox / foreground_mask) to reproduce recon_eval.py's scoring
and break SSIM_bbox down by label x acc and by box area.
"""

import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.common.metrics import SSIM, foreground_mask, ssim_bbox, ssim_full  # noqa: E402

RECON = Path('/root/result/sdum_run10/reconstructions_leaderboard')
DATA = Path('/root/Data/leaderboard')
device = torch.device('cpu')


def run_acc(ssim, acc):
    image_dir = DATA / f'acc{acc}' / 'image'
    recon_dir = RECON / f'acc{acc}'

    full_total, full_idx = 0.0, 0
    bbox_total, bbox_idx = 0.0, 0
    rows = []          # per-box records
    skipped = 0

    for img_path in sorted(image_dir.iterdir()):
        with h5py.File(img_path, 'r') as hf:
            target_vol = hf['image_label'][:]
            maximum = hf.attrs['max']
            annotations = json.loads(hf.attrs.get('annotations', '{}'))
        with h5py.File(recon_dir / img_path.name, 'r') as hf:
            recon_vol = hf['reconstruction'][:]

        assert recon_vol.shape == target_vol.shape, f'{img_path.name}: {recon_vol.shape} vs {target_vol.shape}'
        n = target_vol.shape[0]

        for s in range(n):
            recon_t = torch.from_numpy(recon_vol[s]).to(device)
            target_t = torch.from_numpy(target_vol[s]).to(device)
            mask_t = torch.from_numpy(foreground_mask(target_vol[s])).to(device).type(torch.float)

            slice_full = ssim_full(ssim, recon_t, target_t, mask_t, maximum)
            if slice_full is not None:
                full_total += slice_full
                full_idx += 1

            for box in annotations.get(str(s), []):
                value = ssim_bbox(ssim, recon_t, target_t, box, maximum)
                if value is None:
                    skipped += 1
                    continue
                bbox_total += value
                bbox_idx += 1
                x0, y0 = max(0, box['x']), max(0, box['y'])
                x1 = min(target_t.shape[1], box['x'] + box['width'])
                y1 = min(target_t.shape[0], box['y'] + box['height'])
                # slice_full = how well the model did on the WHOLE slice this box sits in.
                # delta isolates "hard lesion" from "hard slice/volume".
                rows.append(dict(vol=img_path.name, slice=s, acc=acc, label=box['label'],
                                 ssim=value, slice_full=slice_full,
                                 delta=value - slice_full if slice_full is not None else None,
                                 w=x1 - x0, h=y1 - y0, area=(x1 - x0) * (y1 - y0)))

    full_score = full_total / full_idx if full_idx else 0.0
    bbox_score = bbox_total / bbox_idx if bbox_idx else 0.0
    return full_score, bbox_score, full_idx, bbox_idx, rows, skipped


def agg(vals):
    m = statistics.mean(vals)
    sd = statistics.stdev(vals) if len(vals) > 1 else float('nan')
    se = sd / len(vals) ** 0.5 if len(vals) > 1 else float('nan')
    return m, sd, se


if __name__ == '__main__':
    ssim = SSIM().to(device)
    full4, bbox4, nf4, nb4, rows4, sk4 = run_acc(ssim, 4)
    full8, bbox8, nf8, nb8, rows8, sk8 = run_acc(ssim, 8)
    rows = rows4 + rows8

    print('=' * 72)
    print('SANITY CHECK vs run10 scorecard (recon_eval.py aggregation)')
    print('=' * 72)
    print(f'SSIM_full : {(full4 + full8) / 2:.4f}   (expected ~0.9118)')
    print(f'SSIM_bbox : {(bbox4 + bbox8) / 2:.4f}   (expected ~0.9021)')
    print(f'  acc4: full={full4:.4f} (n={nf4} slices)  bbox={bbox4:.4f} (n={nb4} boxes)')
    print(f'  acc8: full={full8:.4f} (n={nf8} slices)  bbox={bbox8:.4f} (n={nb8} boxes)')
    print(f'  boxes skipped by win_size filter: acc4={sk4} acc8={sk8}')

    print('\n' + '=' * 72)
    print('SSIM_bbox by label x acc  (n = number of boxes)')
    print('=' * 72)
    by = defaultdict(list)
    for r in rows:
        by[(r['label'], r['acc'])].append(r['ssim'])
    labels = sorted({r['label'] for r in rows},
                    key=lambda l: -len([r for r in rows if r['label'] == l]))
    print(f'{"label":<45} {"acc4 mean":>10} {"n4":>5} {"acc8 mean":>10} {"n8":>5} {"all mean":>9} {"n":>5} {"sd":>6}')
    for lab in labels:
        v4, v8 = by[(lab, 4)], by[(lab, 8)]
        allv = v4 + v8
        m, sd, _ = agg(allv)
        s4 = f'{statistics.mean(v4):.4f}' if v4 else '   -  '
        s8 = f'{statistics.mean(v8):.4f}' if v8 else '   -  '
        flag = '  <-- small n' if len(allv) < 30 else ''
        print(f'{lab:<45} {s4:>10} {len(v4):>5} {s8:>10} {len(v8):>5} {m:>9.4f} {len(allv):>5} {sd:>6.4f}{flag}')

    print('\n' + '=' * 72)
    print('CONTROL: box SSIM vs the SSIM_full of the slice the box sits in')
    print('delta = box_ssim - slice_full_ssim.  delta<0 => box is HARDER than its own slice.')
    print('A label with low absolute SSIM but delta~0 is a hard-VOLUME artifact, not a hard lesion.')
    print('=' * 72)
    print(f'{"label":<45} {"box":>7} {"slice":>7} {"delta":>8} {"n":>5}')
    for lab in labels:
        sel = [r for r in rows if r['label'] == lab and r['delta'] is not None]
        if not sel:
            continue
        mb = statistics.mean([r['ssim'] for r in sel])
        ms = statistics.mean([r['slice_full'] for r in sel])
        md = statistics.mean([r['delta'] for r in sel])
        flag = '  <-- small n' if len(sel) < 30 else ''
        print(f'{lab:<45} {mb:>7.4f} {ms:>7.4f} {md:>+8.4f} {len(sel):>5}{flag}')
    alld = [r['delta'] for r in rows if r['delta'] is not None]
    print(f'{"ALL BOXES":<45} {statistics.mean([r["ssim"] for r in rows]):>7.4f} '
          f'{statistics.mean([r["slice_full"] for r in rows]):>7.4f} '
          f'{statistics.mean(alld):>+8.4f} {len(alld):>5}')

    print('\n' + '=' * 72)
    print('SSIM_bbox by box area (quintile-ish bins), acc pooled')
    print('=' * 72)
    bins = [(0, 500), (500, 1000), (1000, 2000), (2000, 4000), (4000, 8000), (8000, 10 ** 9)]
    print(f'{"area (px^2)":<16} {"n":>5} {"mean SSIM":>10} {"sd":>7} {"acc4":>8} {"acc8":>8}')
    for lo, hi in bins:
        sel = [r for r in rows if lo <= r['area'] < hi]
        if not sel:
            continue
        m, sd, _ = agg([r['ssim'] for r in sel])
        a4 = [r['ssim'] for r in sel if r['acc'] == 4]
        a8 = [r['ssim'] for r in sel if r['acc'] == 8]
        lbl = f'{lo}-{hi}' if hi < 10 ** 9 else f'{lo}+'
        print(f'{lbl:<16} {len(sel):>5} {m:>10.4f} {sd:>7.4f} '
              f'{statistics.mean(a4) if a4 else float("nan"):>8.4f} '
              f'{statistics.mean(a8) if a8 else float("nan"):>8.4f}')

    areas = np.array([r['area'] for r in rows], dtype=float)
    ss = np.array([r['ssim'] for r in rows], dtype=float)
    print(f'\nPearson corr(area, ssim)      = {np.corrcoef(areas, ss)[0, 1]:+.4f}')
    print(f'Pearson corr(log area, ssim)  = {np.corrcoef(np.log(areas), ss)[0, 1]:+.4f}')
    from scipy.stats import spearmanr
    print(f'Spearman corr(area, ssim)     = {spearmanr(areas, ss).statistic:+.4f}')
    print(f'\nbox area: min={areas.min():.0f} median={np.median(areas):.0f} '
          f'mean={areas.mean():.0f} max={areas.max():.0f}')

    print('\n' + '=' * 72)
    print('worst 15 boxes by SSIM')
    print('=' * 72)
    for r in sorted(rows, key=lambda r: r['ssim'])[:15]:
        print(f'  {r["ssim"]:.4f}  acc{r["acc"]}  {r["vol"]:<22} s={r["slice"]:<3} '
              f'{r["w"]}x{r["h"]}  {r["label"]}')

    out = Path('/tmp/claude-0/-root/13b970be-2cf2-478e-9bd3-fa7a2725f320/scratchpad/bbox_rows.json')
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows))
    print(f'\nper-box records written to {out}')
