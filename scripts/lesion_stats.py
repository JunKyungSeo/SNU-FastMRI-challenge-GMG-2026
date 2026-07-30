"""Stage-0 diagnosis A: count lesion boxes from h5 `attrs['annotations']` only.

CPU-only, reads no pixel data (only attrs + image_label shape for slice counts).
"Valid box" follows utils/common/metrics.py::ssim_bbox exactly: the box is
clipped to the image, and rejected if the clipped width or height < win_size (7).
Slices with index >= num_slices are ignored, matching recon_eval.py:91.
"""

import json
from collections import Counter, defaultdict
from pathlib import Path

import h5py

WIN = 7  # SSIMLoss(win_size=7) default, used by ssim_bbox
IMG = 384

TRAIN = Path('/root/Data/train/image')
LB = [Path('/root/Data/leaderboard/acc4/image'), Path('/root/Data/leaderboard/acc8/image')]


def clipped(box):
    """Return (w, h) of the box after clipping to the image, as ssim_bbox does."""
    x0, y0 = max(0, box['x']), max(0, box['y'])
    x1 = min(IMG, box['x'] + box['width'])
    y1 = min(IMG, box['y'] + box['height'])
    return x1 - x0, y1 - y0


def is_valid(box):
    w, h = clipped(box)
    return w >= WIN and h >= WIN


def scan(dirs, tag):
    n_vol = 0
    n_slices = 0
    n_lesion_slices = 0          # slices with >= 1 valid box
    n_boxes = 0
    n_boxes_raw = 0              # before the win_size filter
    box_count = Counter()        # label -> #valid boxes
    slice_count = defaultdict(set)   # label -> {(vol, slice)}
    vol_count = defaultdict(set)     # label -> {vol}
    per_vol_lesion_slices = {}   # vol -> #lesion slices
    boxes_per_lesion_slice = Counter()   # k -> #slices carrying exactly k valid boxes
    dropped = []                 # boxes rejected by the win_size filter

    for d in dirs:
        for p in sorted(d.iterdir()):
            n_vol += 1
            with h5py.File(p, 'r') as hf:
                n = hf['image_label'].shape[0]
                ann = json.loads(hf.attrs.get('annotations', '{}'))
            n_slices += n
            vol_lesion = 0
            for s in range(n):                      # recon_eval only visits s < n
                boxes = ann.get(str(s), [])
                n_boxes_raw += len(boxes)
                valid = [b for b in boxes if is_valid(b)]
                for b in boxes:
                    if not is_valid(b):
                        w, h = clipped(b)
                        dropped.append((p.name, s, b['label'], b['width'], b['height'], w, h))
                if not valid:
                    continue
                vol_lesion += 1
                n_lesion_slices += 1
                boxes_per_lesion_slice[len(valid)] += 1
                n_boxes += len(valid)
                for b in valid:
                    box_count[b['label']] += 1
                    slice_count[b['label']].add((p.name, s))
                    vol_count[b['label']].add(p.name)
            per_vol_lesion_slices[p.name] = vol_lesion
            # annotation keys pointing past the end of the volume
            for k in ann:
                if int(k) >= n:
                    print(f'  [warn] {p.name}: annotation slice {k} >= num_slices {n} (ignored by recon_eval)')

    return dict(tag=tag, n_vol=n_vol, n_slices=n_slices, n_lesion_slices=n_lesion_slices,
                n_boxes=n_boxes, n_boxes_raw=n_boxes_raw, box_count=box_count,
                slice_count=slice_count, vol_count=vol_count,
                per_vol_lesion_slices=per_vol_lesion_slices,
                boxes_per_lesion_slice=boxes_per_lesion_slice, dropped=dropped)


def report(r):
    print(f"\n{'=' * 70}\n{r['tag']}\n{'=' * 70}")
    p = r['n_lesion_slices'] / r['n_slices'] if r['n_slices'] else 0.0
    print(f"volumes            : {r['n_vol']}")
    print(f"slices (total)     : {r['n_slices']}")
    print(f"lesion slices      : {r['n_lesion_slices']}   (>=1 valid box)")
    print(f"p                  : {p:.6f}   ({p * 100:.3f}%)")
    print(f"valid boxes        : {r['n_boxes']}")
    print(f"raw boxes          : {r['n_boxes_raw']}   (dropped by win_size<{WIN}: {r['n_boxes_raw'] - r['n_boxes']})")

    print('\nlabel frequency (valid boxes only):')
    print(f"{'label':<52} {'boxes':>6} {'slices':>7} {'vols':>5}")
    for lab, c in r['box_count'].most_common():
        print(f'{lab:<52} {c:>6} {len(r["slice_count"][lab]):>7} {len(r["vol_count"][lab]):>5}')

    print('\nboxes per lesion slice:')
    for k in sorted(r['boxes_per_lesion_slice']):
        print(f'  {k} box(es): {r["boxes_per_lesion_slice"][k]} slices')

    ls = sorted(r['per_vol_lesion_slices'].values())
    zero = sum(1 for v in ls if v == 0)
    print('\nlesion slices per volume:')
    print(f'  volumes with 0 lesion slices: {zero} / {r["n_vol"]}')
    if ls:
        import statistics
        print(f'  min={ls[0]}  median={statistics.median(ls)}  mean={statistics.mean(ls):.2f}  max={ls[-1]}')
        print(f'  distribution: {dict(sorted(Counter(ls).items()))}')

    if r['dropped']:
        print(f'\ndropped boxes (clipped w or h < {WIN}):')
        for d in r['dropped']:
            print(f'  {d[0]} s={d[1]} {d[2]!r} raw={d[3]}x{d[4]} clipped={d[5]}x{d[6]}')


if __name__ == '__main__':
    tr = scan([TRAIN], 'TRAIN  (/root/Data/train/image)')
    lb = scan(LB, 'LEADERBOARD  (acc4 + acc8)')
    report(tr)
    report(lb)

    print(f"\n{'=' * 70}\np and oversampling weights\n{'=' * 70}")
    p = tr['n_lesion_slices'] / tr['n_slices']
    print(f'p = {p:.6f}')
    print(f'{"target q":>10} {"w = q(1-p)/(p(1-q))":>22}')
    for m in (1, 2, 3):
        q = m * p
        w = q * (1 - p) / (p * (1 - q))
        print(f'{q:>10.6f} {w:>22.4f}   (q = {m}p)')
