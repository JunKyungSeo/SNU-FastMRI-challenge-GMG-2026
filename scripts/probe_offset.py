"""마스크 offset 민감도 탐침 — 학습 0, 기존 체크포인트로 채점만.

학습풀 offset은 (W//2)%acc 로 결정론이고 채점은 무작위 0~acc-1 이라, acc8 전문가는
채점 볼륨의 65%를 학습에서 본 적 없는 offset 으로 만난다(strategy/augmentation.md).
이 스크립트는 **그 격차가 실제로 점수를 얼마나 깎는지**를 잰다 — hold-out 볼륨의 마스크만
offset 0..acc-1 로 갈아끼우고 같은 모델로 복원해 SSIM_full/SSIM_bbox 를 offset 별로 찍는다.

핵심: canonical(=학습에서 본 offset) 과 나머지의 차이가 `--remask` 가 회수할 수 있는 상한이다.
차이가 없으면 모델이 offset 에 둔감하다는 뜻이라 remask 본실험을 접을 근거가 된다.

추론 경로는 test_part.recon_slice 를 그대로 재사용한다(readout_crop·coil_compress 재적용까지
동일) — 마스크만 우리가 만든 것으로 바꿔 ctx 에 넣는다.

실행: python scripts/probe_offset.py -n promptmr_moe_acc8 -p /root/Data/val --acc 8
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import torch

# scripts/ 에서 실행되므로 sys.path[0] 은 scripts/ 다 — repo 루트를 직접 넣어야 utils 가 보인다
# (recon_eval.py 는 루트에 있어서 이게 필요 없다). utils/model 은 varnet 의 `import fastmri` 용.
_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_ROOT), str(_ROOT / "utils" / "model")):
    if _p not in sys.path:
        sys.path.insert(1, _p)

from utils.common.metrics import SSIM, foreground_mask, ssim_bbox, ssim_full
from utils.data.load_data import KFOLD_ROOT_SEP, kfold_split
from utils.data.transforms import canonical_offset, equispaced_mask
from utils.learning.test_part import load_model, recon_slice


def parse():
    p = argparse.ArgumentParser(description="마스크 offset 민감도 탐침 (학습 없음)")
    p.add_argument('-g', '--GPU_NUM', type=int, default=0)
    p.add_argument('-n', '--net_name', type=str, required=True,
                   help='체크포인트 이름 (../result/<net>/checkpoints/best_model.pt)')
    p.add_argument('-p', '--path_data', type=Path, default=Path('/root/Data/val'),
                   help='고정분할 모델용 hold-out 폴더(image/·kspace/). --k-fold를 주면 무시된다.')
    p.add_argument('--acc', type=int, default=8, choices=[4, 8],
                   help='이 가속률 볼륨만 본다. 전문가의 acc 와 맞출 것.')
    # k-fold 로 학습된 모델은 hold-out 이 폴더가 아니라 fold 다 — 학습 때와 **같은 값**을 줘야
    # 진짜 hold-out 이 된다(어긋나면 학습에 쓴 볼륨으로 재는 셈이라 격차가 과소평가된다).
    p.add_argument('--k-fold', type=int, default=0, help='>0이면 k-fold hold-out 을 쓴다(폴더 대신).')
    p.add_argument('--val-fold', type=int, default=0)
    p.add_argument('--kfold-seed', type=int, default=430)
    p.add_argument('--data-path-train', type=Path, default=Path('/root/Data/train/'))
    p.add_argument('--data-path-val', type=Path, default=Path('/root/Data/val/'))
    p.add_argument('--max-volumes', type=int, default=0, help='볼륨 수 제한 (0=전부). 스모크용.')
    p.add_argument('--max-slices', type=int, default=0, help='볼륨당 슬라이스 제한 (0=전부). 스모크용.')
    # load_model 이 옛 VarNet 경로에서 참조하는 값들(promptmr 체크포인트면 저장 args 가 이긴다).
    p.add_argument('--cascade', type=int, default=1)
    p.add_argument('--chans', type=int, default=9)
    p.add_argument('--sens_chans', type=int, default=4)
    return p.parse_args()


def score_volume(model, ssim, device, img_path, ks_path, acc, max_slices):
    """한 볼륨을 offset 0..acc-1 각각으로 복원해 (offset -> [full 합, full 수, bbox 합, bbox 수]) 반환."""
    with h5py.File(img_path, 'r') as hf:
        target_vol = hf['image_label'][:]
        maximum = hf.attrs['max']
        annotations = json.loads(hf.attrs.get('annotations', '{}'))
    with h5py.File(ks_path, 'r') as hf:
        kspace = hf['kspace'][:]
        mask_dtype = np.array(hf['mask']).dtype

    n = kspace.shape[0] if max_slices <= 0 else min(max_slices, kspace.shape[0])
    W = kspace.shape[-1]
    canon = canonical_offset(W, acc)

    out = {}
    for off in range(acc):
        m = equispaced_mask(W, acc, off, dtype=mask_dtype)
        ctx = {"kspace": kspace, "mask": m, "device": device, "num_slices": n}
        f_sum, f_cnt, b_sum, b_cnt = 0.0, 0, 0.0, 0
        with torch.no_grad():
            for s in range(n):
                recon_t = recon_slice(model, ctx, s)
                target_t = torch.from_numpy(target_vol[s]).to(device=device)
                fg_t = torch.from_numpy(foreground_mask(target_vol[s])).to(device=device).float()
                v = ssim_full(ssim, recon_t, target_t, fg_t, maximum)
                if v is not None:
                    f_sum += v
                    f_cnt += 1
                for box in annotations.get(str(s), []):
                    v = ssim_bbox(ssim, recon_t, target_t, box, maximum)
                    if v is not None:
                        b_sum += v
                        b_cnt += 1
        out[off] = (f_sum, f_cnt, b_sum, b_cnt)
    return out, canon, W


def holdout_volumes(args):
    """(image_path, kspace_path) 목록. k-fold면 그 fold의 hold-out, 아니면 --path_data 폴더."""
    if args.k_fold > 0:
        _, val_ids = kfold_split(args)   # --acc(int)는 _norm_acc가 str로 정규화한다
        roots = {Path(args.data_path_train).name: Path(args.data_path_train),
                 Path(args.data_path_val).name: Path(args.data_path_val)}
        out = []
        for vid in sorted(val_ids):
            root_name, name = vid.split(KFOLD_ROOT_SEP, 1)
            r = roots[root_name]
            if f'acc{args.acc}' in name and (r / 'image' / name).exists():
                out.append((r / 'image' / name, r / 'kspace' / name))
        return out
    img_dir, ks_dir = args.path_data / 'image', args.path_data / 'kspace'
    assert img_dir.is_dir() and ks_dir.is_dir(), f"{args.path_data} 에 image/ kspace/ 가 없다"
    return [(p, ks_dir / p.name) for p in sorted(img_dir.iterdir())
            if p.suffix == '.h5' and f'acc{args.acc}' in p.name and (ks_dir / p.name).exists()]


if __name__ == '__main__':
    args = parse()
    args.exp_dir = Path('../result') / args.net_name / 'checkpoints'

    device = torch.device(f'cuda:{args.GPU_NUM}' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.cuda.set_device(device)

    model = load_model(args, device)
    ssim = SSIM().to(device=device)

    pairs = holdout_volumes(args)
    if args.max_volumes > 0:
        pairs = pairs[:args.max_volumes]
    assert pairs, f"acc{args.acc} hold-out 볼륨을 못 찾았다"

    src = (f"k-fold {args.k_fold} fold{args.val_fold} seed{args.kfold_seed}"
           if args.k_fold > 0 else str(args.path_data))
    print(f"[probe] net={args.net_name}  acc={args.acc}  hold-out={src}  "
          f"볼륨 {len(pairs)}개  offset 0~{args.acc-1}")

    agg = defaultdict(lambda: [0.0, 0, 0.0, 0])     # offset -> 전역 누산기
    canon_agg = [0.0, 0, 0.0, 0]                     # 각 볼륨의 canonical offset 만 모은 것
    other_agg = [0.0, 0, 0.0, 0]                     # 그 외 offset
    canon_seen = defaultdict(int)

    for i, (img_path, ks_path) in enumerate(pairs, 1):
        per_off, canon, W = score_volume(
            model, ssim, device, img_path, ks_path, args.acc, args.max_slices)
        canon_seen[canon] += 1
        for off, (fs, fc, bs, bc) in per_off.items():
            a = agg[off]
            a[0] += fs; a[1] += fc; a[2] += bs; a[3] += bc
            tgt = canon_agg if off == canon else other_agg
            tgt[0] += fs; tgt[1] += fc; tgt[2] += bs; tgt[3] += bc
        print(f"  [{i}/{len(pairs)}] {img_path.name}  W={W}  canonical_offset={canon}")

    def mean(a):
        return (a[0] / a[1] if a[1] else 0.0), (a[2] / a[3] if a[3] else 0.0)

    print("\n===== offset 별 =====")
    print(f"{'offset':>7} {'SSIM_full':>11} {'SSIM_bbox':>11}   비고")
    for off in range(args.acc):
        f, b = mean(agg[off])
        tag = f"학습에서 봄 (볼륨 {canon_seen[off]}개의 canonical)" if canon_seen.get(off) else ""
        print(f"{off:>7} {f:>11.4f} {b:>11.4f}   {tag}")

    cf, cb = mean(canon_agg)
    of, ob = mean(other_agg)
    print("\n===== 판정 =====")
    print(f"canonical(학습에서 본 offset) : full {cf:.4f}  bbox {cb:.4f}")
    print(f"그 외 offset                  : full {of:.4f}  bbox {ob:.4f}")
    print(f"격차                          : full {cf - of:+.4f}  bbox {cb - ob:+.4f}")
    gap = max(cf - of, cb - ob)
    if gap < 0.002:
        print("→ 모델이 offset 에 둔감하다. --remask 로 회수할 게 거의 없다.")
    elif gap < 0.005:
        print("→ 격차가 작다. --remask 이득도 그 이하일 것.")
    else:
        print("→ 격차가 실재한다. --remask 본실험 근거가 된다(회수 상한 ≈ 이 격차).")
