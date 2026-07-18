"""recon_eval 래퍼 — 채점 결과(SSIM_full·SSIM_bbox·시간)를 리포트 파일로 저장.

recon_eval.py는 **고정(수정 금지)이라 stdout에 print만** 한다. 이 래퍼는 recon_eval의
채점 함수(run_acc)를 그대로 재사용해 동일 점수를 산출한 뒤, run이 끝나면 구조화된
스코어카드에 **한 줄 append**한다(sweep = 표). recon_eval.py 자체는 건드리지 않는다.

사용(인자는 recon_eval와 동일 + 기록용):
    python scripts/recon_eval_report.py -n <net_name> -p /root/Data/leaderboard \
        [--n-virtual-coils 8] [--note "run13 nvc8"]
출력:
    ../result/<net_name>/recon_eval_scorecard.jsonl   (한 줄=한 eval, append)
    + stdout는 recon_eval와 동일하게 출력
"""
import argparse
import datetime
import json
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, os.getcwd())
if os.getcwd() + '/utils/model/' not in sys.path:
    sys.path.insert(1, os.getcwd() + '/utils/model/')

import recon_eval as RE   # 고정 하네스: run_acc / load_model / SSIM 재사용(무수정)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('-g', '--GPU_NUM', type=int, default=0)
    ap.add_argument('-n', '--net_name', type=Path, default='test_varnet')
    ap.add_argument('-p', '--path_data', type=Path, default='/Data/leaderboard/')
    # varnet 폴백 대비(promptmr는 저장 args에서 복원되어 미사용). recon_eval와 parity.
    ap.add_argument('--cascade', type=int, default=1)
    ap.add_argument('--chans', type=int, default=9)
    ap.add_argument('--sens_chans', type=int, default=4)
    ap.add_argument('--input_key', type=str, default='kspace')
    ap.add_argument('--note', type=str, default='', help='스코어카드에 남길 자유 메모')
    ap.add_argument('-o', '--out', type=Path, default=None,
                    help='스코어카드 경로(기본 ../result/<net>/recon_eval_scorecard.jsonl)')
    args = ap.parse_args()

    args.exp_dir = Path('../result') / args.net_name / 'checkpoints'
    assert (args.path_data / 'acc4').is_dir() and (args.path_data / 'acc8').is_dir(), \
        f"acc4/acc8 폴더 필요: {args.path_data}"

    device = torch.device(f'cuda:{args.GPU_NUM}' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.cuda.set_device(device)

    model = RE.load_model(args, device)
    # 실제 적용된 압축 설정을 모델에서 읽음(ckpt saved args에서 복원됨). 0 = 압축 없음.
    n_virtual = getattr(model, "n_virtual_coils", 0) if getattr(model, "coil_compress", False) else 0
    ssim = RE.SSIM().to(device=device)
    recon_base = Path('../result') / args.net_name / 'reconstructions_leaderboard'

    full4, bbox4, t4, m4, slow4, n4 = RE.run_acc(model, ssim, device, args.path_data / 'acc4', recon_base / 'acc4')
    full8, bbox8, t8, m8, slow8, n8 = RE.run_acc(model, ssim, device, args.path_data / 'acc8', recon_base / 'acc8')

    total_time = t4 + t8
    total_slices = n4 + n8
    total_ms = total_time * 1e3 / total_slices if total_slices > 0 else 0.0
    ssim_full = (full4 + full8) / 2
    ssim_bbox = (bbox4 + bbox8) / 2

    # ── stdout: recon_eval와 동일 ──
    print("Leaderboard SSIM_full : {:.4f}".format(ssim_full))
    print("Leaderboard SSIM_bbox : {:.4f}".format(ssim_bbox))
    print("Leaderboard Recon Time : {:.2f}s ({:.1f} ms/slice)".format(total_time, total_ms))
    print("SSIM_full (acc4/8): {:.4f} / {:.4f}   SSIM_bbox (acc4/8): {:.4f} / {:.4f}".format(
        full4, full8, bbox4, bbox8))
    if slow4 + slow8 > 0:
        print("[WARNING] {} volume(s) exceeded {:.0f}s forward time".format(slow4 + slow8, RE.VOLUME_TIME_WARN_S))

    # ── 스코어카드 append(한 줄=한 eval) ──
    row = {
        "time": datetime.datetime.now().isoformat(timespec="seconds"),
        "net_name": str(args.net_name),
        "n_virtual_coils": n_virtual,
        "ssim_full": round(ssim_full, 4), "ssim_bbox": round(ssim_bbox, 4),
        "full_acc4": round(full4, 4), "full_acc8": round(full8, 4),
        "bbox_acc4": round(bbox4, 4), "bbox_acc8": round(bbox8, 4),
        "recon_time_s": round(total_time, 2), "ms_per_slice": round(total_ms, 1),
        "note": args.note,
    }
    out = args.out or (Path('../result') / args.net_name / 'recon_eval_scorecard.jsonl')
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'a') as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[recon_eval_report] appended → {out}")


if __name__ == '__main__':
    main()
