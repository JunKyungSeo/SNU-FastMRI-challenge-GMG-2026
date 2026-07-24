#!/usr/bin/env python3
"""K-Fold의 hold-out fold를 recon_eval.py가 읽는 leaderboard 형식으로 심볼릭 링크한다.

[왜 필요한가]
  이 서버엔 leaderboard 데이터가 없다(/root/Data/leaderboard 부재). 그래서 학습 뒤 채점 단계가
  recon_eval.py:120의 AssertionError로 죽는다 — promptmr_run4도, promptmr_kfold_f1도 21시간
  학습을 마치고 여기서 실패해 점수를 한 줄도 못 남겼다.
  K-Fold의 hold-out fold는 image_label과 attrs['annotations']를 모두 갖고 있어(200/200 volume 확인)
  그대로 채점 세트가 된다 → recon_eval.py(고정·수정 금지)를 건드리지 않고 -p 만 이 디렉터리로
  돌리면 SSIM_full·SSIM_bbox·추론시간 3축이 그대로 나온다.

[출력 구조]  recon_eval.py가 요구하는 acc4/acc8 × image/kspace
    <out>/acc4/{image,kspace}/<volume_id>      # volume_id 예: train__knee_acc4_1.h5
    <out>/acc8/{image,kspace}/<volume_id>
  파일명에 root 태그(train/val)를 붙이는 이유는 train/과 val/에 이름이 같은 다른 볼륨이
  30쌍 있어서다(utils/data/load_data.volume_id 참고).

[주의] hold-out fold 점수는 학습에 쓰지 않은 데이터라 정직하지만, leaderboard 점수와 같지는 않다
  (분포·박스 수가 다르다). fold끼리, run끼리 비교하는 용도로 쓸 것.

사용:
    python scripts/make_fold_eval_dir.py --k-fold 5 --val-fold 2 -o ../result/kfold_eval/f2
    python recon_eval.py -n promptmr_kfold_f2 -p ../result/kfold_eval/f2
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.data.load_data import KFOLD_ROOT_SEP, kfold_roots, kfold_split   # noqa: E402


def parse():
    p = argparse.ArgumentParser(
        description='hold-out fold를 recon_eval용 leaderboard 형식으로 링크',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('-t', '--data-path-train', type=Path, default=Path('/workspace/Data/train/'))
    p.add_argument('-v', '--data-path-val', type=Path, default=Path('/workspace/Data/val/'))
    p.add_argument('--k-fold', type=int, default=5, help='분할 수 (train.py --k-fold와 같아야 함)')
    p.add_argument('--val-fold', type=int, default=0, help='hold-out fold 인덱스 (train.py --val-fold와 같아야 함)')
    p.add_argument('--kfold-seed', type=int, default=430, help='분할 seed (train.py --kfold-seed와 같아야 함)')
    p.add_argument('-o', '--out', type=Path, required=True, help='생성할 채점 디렉터리')
    return p.parse_args()


def main():
    args = parse()
    _, val_ids = kfold_split(args)   # 학습과 같은 함수 → 분할이 어긋날 수 없다
    roots = {r.name: r for r in kfold_roots(args)}

    # 같은 out 디렉터리를 다른 fold로 재사용하면 이전 fold의 링크가 남아 채점 세트가 오염된다.
    # 우리가 만든 심볼릭 링크만 지운다 — 실제 데이터 파일은 절대 건드리지 않는다.
    stale = 0
    for acc in ('acc4', 'acc8'):
        for kind in ('image', 'kspace'):
            d = args.out / acc / kind
            if d.is_dir():
                for f in d.iterdir():
                    if f.is_symlink():
                        f.unlink()
                        stale += 1

    n = {'acc4': 0, 'acc8': 0}
    for vol_id in sorted(val_ids):
        tag, name = vol_id.split(KFOLD_ROOT_SEP, 1)
        acc = 'acc4' if 'acc4' in name else 'acc8'
        for kind in ('image', 'kspace'):
            dst_dir = args.out / acc / kind
            dst_dir.mkdir(parents=True, exist_ok=True)
            (dst_dir / vol_id).symlink_to((roots[tag] / kind / name).resolve())
        n[acc] += 1

    print(f"[fold-eval] {args.out}  (fold {args.val_fold}/{args.k_fold}, seed {args.kfold_seed})")
    if stale:
        print(f"[fold-eval] 이전 링크 {stale}개 정리")
    print(f"[fold-eval] hold-out volume {len(val_ids)}개 링크: acc4 {n['acc4']} · acc8 {n['acc8']}")


if __name__ == '__main__':
    main()
