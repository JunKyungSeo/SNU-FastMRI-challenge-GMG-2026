"""방향(orientation) 민감도 탐침 — 학습 0, 기존 체크포인트로 채점만.

180° 회전을 증강에 넣을지 정하려고 만들었다. hold-out 볼륨을 네 방향
(원본 / 좌우 flip / 상하 flip / 180°)으로 각각 복원해 SSIM_full·SSIM_bbox 를 찍는다.
k-space·마스크·target·bbox 를 함께 돌리므로 정답 정렬은 유지된다 — 즉 "같은 무릎을
뒤집어 찍은 획득"을 모델이 얼마나 못 푸는지를 재는 것이다.

변환은 전부 닫힌 형태다(보간 없음, FFT 없음). transforms._flip_lr 과 같은 계열:
  좌우: K → roll(flip(K,-1),1,-1)·ramp_w,  mask → roll(flip(M),1)   (짝수 W 전제)
  상하: K → roll(flip(K,-2),1,-2)·ramp_h,  mask 불변                 (짝수 H 전제)
샘플 볼륨 실측 relmaxerr — 좌우 2.5e-07 · 상하 1.6e-07 · 180° 2.2e-07.
마스크는 폭 방향 1D 라 상하 반전과 무관하고, 좌우 반전은 라인 수를 보존하므로 MoE 라우터의
샘플 비율(임계 0.25)도 안 흔들린다. 640→384 readout crop 은 128/128 대칭이라 상하 반전과 교환된다.

읽는 법: 격차(원본 − 뒤집은 것)가 그 방향 증강이 **새로 가르칠 수 있는 양의 상한**이다.
  격차 ≈ 0 → 모델이 이미 그 방향에 둔감하다. 증강해도 배울 게 없으니 접는다.
  격차 큼  → 배울 여지는 있다. 다만 채점 데이터엔 뒤집힌 무릎이 없으므로 "리더보드 점수가
             오른다"는 뜻이 아니다 — A/B 를 태울 근거가 될 뿐이다.
  lr 은 대조군이다. flip 증강으로 학습된 체크포인트면 0 에 가까워야 정상이고, 크게 나오면
  증강이 실제로는 안 걸렸다는 신호다. 180° 격차 ≈ 상하 격차면 새 정보는 전부 상하 성분이다.

실행: python scripts/probe_orient.py -n promptmr_moe_f2_shared_aug --acc 8 \
          --k-fold 5 --val-fold 2 --kfold-seed 430
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import torch

# scripts/ 에서 실행되므로 sys.path[0] 은 scripts/ 다 — repo 루트를 직접 넣어야 utils 가 보인다.
_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_ROOT), str(_ROOT / "utils" / "model")):
    if _p not in sys.path:
        sys.path.insert(1, _p)

from probe_offset import holdout_volumes          # hold-out 목록 규칙은 offset 탐침과 동일
from utils.common.metrics import SSIM, foreground_mask, ssim_bbox, ssim_full
from utils.learning.test_part import load_model, recon_slice

ORIENTS = ("id", "lr", "ud", "rot180")
LABEL = {"id": "원본", "lr": "좌우flip", "ud": "상하flip", "rot180": "180°"}


def parse():
    p = argparse.ArgumentParser(description="방향(좌우/상하/180°) 민감도 탐침 (학습 없음)")
    p.add_argument('-g', '--GPU_NUM', type=int, default=0)
    p.add_argument('-n', '--net_name', type=str, required=True,
                   help='체크포인트 이름 (../result/<net>/checkpoints/best_model.pt)')
    p.add_argument('-p', '--path_data', type=Path, default=Path('/root/Data/val'),
                   help='고정분할 모델용 hold-out 폴더(image/·kspace/). --k-fold를 주면 무시된다.')
    p.add_argument('--acc', type=int, default=8, choices=[4, 8])
    # k-fold 로 학습된 모델은 hold-out 이 폴더가 아니라 fold 다 — 학습 때와 **같은 값**을 줄 것.
    p.add_argument('--k-fold', type=int, default=0, help='>0이면 k-fold hold-out 을 쓴다(폴더 대신).')
    p.add_argument('--val-fold', type=int, default=0)
    p.add_argument('--kfold-seed', type=int, default=430)
    p.add_argument('--data-path-train', type=Path, default=Path('/root/Data/train/'))
    p.add_argument('--data-path-val', type=Path, default=Path('/root/Data/val/'))
    p.add_argument('--max-volumes', type=int, default=0, help='볼륨 수 제한 (0=전부).')
    p.add_argument('--max-slices', type=int, default=0, help='볼륨당 슬라이스 제한 (0=전부).')
    # load_model 이 옛 VarNet 경로에서 참조하는 값들(promptmr 체크포인트면 저장 args 가 이긴다).
    p.add_argument('--cascade', type=int, default=1)
    p.add_argument('--chans', type=int, default=9)
    p.add_argument('--sens_chans', type=int, default=4)
    return p.parse_args()


def orient_kspace(ks, mask, mode):
    """(coils,H,W) complex + (W,) mask 를 mode 방향으로. 보간·FFT 없음."""
    if mode in ("lr", "rot180"):
        W = ks.shape[-1]
        assert W % 2 == 0, f"좌우 반전 공식 roll(flip(·),1)은 짝수 폭에서만 성립한다 (W={W})"
        ramp = np.exp(2j * np.pi * (np.arange(W) - W // 2) / W).astype(ks.dtype)
        ks = np.roll(np.flip(ks, -1), 1, -1) * ramp
        mask = np.roll(np.flip(mask), 1)
    if mode in ("ud", "rot180"):
        H = ks.shape[-2]
        assert H % 2 == 0, f"상하 반전 공식은 짝수 높이에서만 성립한다 (H={H})"
        ramp = np.exp(2j * np.pi * (np.arange(H) - H // 2) / H).astype(ks.dtype)[:, None]
        ks = np.roll(np.flip(ks, -2), 1, -2) * ramp
    return np.ascontiguousarray(ks), np.ascontiguousarray(mask)


def orient_target(t, mode):
    if mode in ("lr", "rot180"):
        t = t[:, ::-1]
    if mode in ("ud", "rot180"):
        t = t[::-1, :]
    return np.ascontiguousarray(t)


def orient_boxes(boxes, mode, size=384):
    """annotation 은 384² 좌표계 — 기하 변환을 박스에도 똑같이 먹인다(안 먹이면 엉뚱한 데를 잰다)."""
    out = []
    for b in boxes:
        b = dict(b)
        if mode in ("lr", "rot180"):
            b["x"] = size - int(b["x"]) - int(b["width"])
        if mode in ("ud", "rot180"):
            b["y"] = size - int(b["y"]) - int(b["height"])
        out.append(b)
    return out


def score_volume(model, ssim, device, img_path, ks_path, max_slices):
    """한 볼륨을 네 방향으로 복원해 mode -> [full 합, full 수, bbox 합, bbox 수] 반환."""
    with h5py.File(img_path, 'r') as hf:
        target_vol = hf['image_label'][:]
        maximum = hf.attrs['max']            # label 의 성질이라 기하 변환과 무관
        annotations = json.loads(hf.attrs.get('annotations', '{}'))
    with h5py.File(ks_path, 'r') as hf:
        kspace = hf['kspace'][:]
        mask = np.array(hf['mask'])

    n = kspace.shape[0] if max_slices <= 0 else min(max_slices, kspace.shape[0])
    out = {}
    for mode in ORIENTS:
        f_sum, f_cnt, b_sum, b_cnt = 0.0, 0, 0.0, 0
        with torch.no_grad():
            for s in range(n):
                # 볼륨 전체가 아니라 슬라이스 1장만 변환한다(볼륨 통째면 사본이 수백 MB).
                ks, m = orient_kspace(kspace[s], mask, mode)
                ctx = {"kspace": ks[None], "mask": m, "device": device, "num_slices": 1}
                recon_t = recon_slice(model, ctx, 0)
                tgt = orient_target(target_vol[s], mode)
                target_t = torch.from_numpy(tgt).to(device=device)
                fg_t = torch.from_numpy(foreground_mask(tgt)).to(device=device).float()
                v = ssim_full(ssim, recon_t, target_t, fg_t, maximum)
                if v is not None:
                    f_sum += v
                    f_cnt += 1
                for box in orient_boxes(annotations.get(str(s), []), mode):
                    v = ssim_bbox(ssim, recon_t, target_t, box, maximum)
                    if v is not None:
                        b_sum += v
                        b_cnt += 1
        out[mode] = (f_sum, f_cnt, b_sum, b_cnt)
    return out


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
          f"볼륨 {len(pairs)}개 × 방향 {len(ORIENTS)}개")

    agg = defaultdict(lambda: [0.0, 0, 0.0, 0])
    for i, (img_path, ks_path) in enumerate(pairs, 1):
        per_mode = score_volume(model, ssim, device, img_path, ks_path, args.max_slices)
        for mode, (fs, fc, bs, bc) in per_mode.items():
            a = agg[mode]
            a[0] += fs; a[1] += fc; a[2] += bs; a[3] += bc
        print(f"  [{i}/{len(pairs)}] {img_path.name}")

    def mean(a):
        return (a[0] / a[1] if a[1] else 0.0), (a[2] / a[3] if a[3] else 0.0)

    base_f, base_b = mean(agg["id"])
    print("\n===== 방향별 =====")
    print(f"{'방향':>9} {'SSIM_full':>11} {'SSIM_bbox':>11} {'Δfull':>9} {'Δbbox':>9}")
    gaps = {}
    for mode in ORIENTS:
        f, b = mean(agg[mode])
        if mode == "id":
            print(f"{LABEL[mode]:>9} {f:>11.4f} {b:>11.4f} {'-':>9} {'-':>9}")
            continue
        gaps[mode] = (base_f - f, base_b - b)
        print(f"{LABEL[mode]:>9} {f:>11.4f} {b:>11.4f} {base_f - f:>+9.4f} {base_b - b:>+9.4f}")

    print("\n===== 판정 =====")
    lr_gap = max(gaps["lr"])
    print(f"[대조군] 좌우 격차 {lr_gap:+.4f} — flip 증강으로 학습된 체크포인트면 0 근처여야 한다."
          + ("  ✓" if abs(lr_gap) < 0.002 else "  ⚠️ 크다: 증강이 실제로 안 걸렸는지 확인할 것"))
    ud_gap, r180_gap = max(gaps["ud"]), max(gaps["rot180"])
    print(f"[신규분] 상하 {ud_gap:+.4f} · 180° {r180_gap:+.4f} "
          f"(180° 격차 ≈ 상하 격차면 새 정보는 전부 상하 성분이다)")
    if r180_gap < 0.002:
        print("→ 모델이 이미 180° 에 둔감하다. 증강해도 새로 가르칠 게 없으니 **접는다**.")
    elif r180_gap < 0.010:
        print("→ 격차가 작다. 180° 증강이 회수할 수 있는 상한이 이 값이라, A/B 60h 를 태울 값은 아니다.")
    else:
        print("→ 격차가 크다. 배울 여지는 실재한다 — 다만 채점 데이터엔 뒤집힌 무릎이 없으므로")
        print("   이 값이 리더보드 이득으로 그대로 바뀌지는 않는다. A/B 근거로만 쓸 것.")
    print("⚠️ 이건 '증강이 새로 가르칠 수 있는 양의 상한'이지 점수 예측이 아니다.")
