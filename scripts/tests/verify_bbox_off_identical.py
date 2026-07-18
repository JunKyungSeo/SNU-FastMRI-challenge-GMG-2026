"""C1 off(λ=0) 비트 동일성 실측 — CPU only, GPU/실모델 불필요.

git HEAD 의 **원본** train_epoch 과 C1 이 들어간 **현재** train_epoch 을,
같은 seed·같은 데이터·같은 모델로 돌려 iter별 loss 를 비교한다.
(--fg-loss 선례와 같은 방식: 끄면 기존 학습과 완전히 동일해야 한다.)

실행: CUDA_VISIBLE_DEVICES="" python scripts/verify_bbox_off_identical.py
"""
import os
import subprocess
import sys
import types
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.append(str(ROOT / "utils" / "model"))   # vendored fastmri (varnet.py:11) — conftest.py와 동일

# GPU 없이 train_epoch 을 돌리기 위해 .cuda() 를 no-op 으로 (수치 경로는 그대로 CPU float32).
torch.Tensor.cuda = lambda self, *a, **k: self

from utils.common.loss_function import SSIMLoss          # noqa: E402
from utils.data.load_data import create_data_loaders     # noqa: E402


class TinyNet(nn.Module):
    """(kspace, mask) → (B,384,384). 실모델 대신 결정적·미분가능한 최소 대체물."""

    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.tensor(0.5))

    def forward(self, kspace, mask):
        x = torch.view_as_complex(kspace.contiguous()).abs()      # (B,C,H,W)
        x = x.mean(1)                                             # coil 평균 → (B,H,W)
        x = torch.nn.functional.interpolate(
            x[:, None], size=(384, 384), mode="bilinear", align_corners=False)[:, 0]
        return self.w * x


def load_original_train_epoch():
    """git HEAD 의 train_part.py 를 임시 모듈로 로드해 원본 train_epoch 을 얻는다."""
    src = subprocess.run(["git", "show", "HEAD:utils/learning/train_part.py"],
                         cwd=ROOT, capture_output=True, text=True, check=True).stdout
    mod = types.ModuleType("train_part_head")
    mod.__dict__["__file__"] = "train_part_head.py"
    exec(compile(src, "train_part_HEAD.py", "exec"), mod.__dict__)
    return mod.train_epoch


def make_args(**over):
    args = types.SimpleNamespace(
        max_key="max", target_key="image_label", input_key="kspace",
        batch_size=1, num_workers=0, seed=430,
        readout_crop=True, crop_h=384,          # run10/11 설정
        # knee_acc4_1.h5 는 slice 13·14 에 박스가 있다 → 20 iter 면 '박스 없는 슬라이스'와
        # '박스 있는 슬라이스'를 모두 통과한다(둘 다 검증해야 의미가 있음).
        max_iters=20, report_interval=10_000, num_epochs=1,
        bbox_loss_lambda=0.0, return_bbox=False, fg_loss=False,
    )
    for k, v in over.items():
        setattr(args, k, v)
    return args


def run(train_epoch_fn, args, tag):
    torch.manual_seed(430)
    model = TinyNet()
    opt = torch.optim.SGD(model.parameters(), lr=1e-3)
    loader = create_data_loaders(data_path=Path("/root/Data/train/"), args=args, shuffle=False,
                                 is_train=True)          # shuffle=False → 순서 고정(비교 가능)
    loss_type = SSIMLoss()

    seen = []
    orig_step = opt.step

    def spy_step():
        orig_step()
    opt.step = spy_step

    # train_epoch 내부 loss.item() 을 못 보므로, loss_type 을 감싸 iter별 값을 기록
    class Spy(nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner
            self.win_size = inner.win_size

        def forward(self, *a, **k):
            out = self.inner(*a, **k)
            # bbox_ssim_loss 도 이 모듈을 호출한다(crop 입력). 전체 384² 호출만 = SSIM_full 항.
            if a[0].shape[-2:] == (384, 384):
                seen.append(float(out.detach()))
            return out
    spy = Spy(loss_type)
    spy.w = loss_type.w
    total, _ = train_epoch_fn(args, 0, model, loader, opt, spy)
    print(f"  {tag}: total={total!r}")
    return total, seen


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    original = load_original_train_epoch()
    from utils.learning.train_part import train_epoch as current

    print("[A] git HEAD 원본 train_epoch (C1 없음)")
    tot_a, losses_a = run(original, make_args(), "HEAD")

    print("[B] 현재 train_epoch, --bbox-loss-lambda 0.0 (=off)")
    tot_b, losses_b = run(current, make_args(bbox_loss_lambda=0.0), "C1-off")

    print("[C] 현재 train_epoch, --bbox-loss-lambda 0.1 (=on)  ← 달라져야 정상")
    tot_c, losses_c = run(current, make_args(bbox_loss_lambda=0.1, return_bbox=True), "C1-on")

    print()
    print("iter별 loss (SSIM_full 항):")
    for i, (a, b, c) in enumerate(zip(losses_a, losses_b, losses_c)):
        print(f"  iter{i}  HEAD={a:.12f}  off={b:.12f}  on={c:.12f}   "
              f"{'IDENTICAL' if a == b else 'DIFFER'}")

    same = (tot_a == tot_b) and all(a == b for a, b in zip(losses_a, losses_b))
    print()
    print(f"  epoch total : HEAD={tot_a!r}")
    print(f"                off ={tot_b!r}")
    print(f"  → off 비트 동일: {'✅ PASS (모든 값 == 완전 일치)' if same else '❌ FAIL'}")
    if tot_a == tot_c:
        print("  ⚠️  on(λ=0.1)이 HEAD와 같다 = λ항이 손실에 안 들어갔다는 뜻 → 배선 실패")
    else:
        print(f"  → on(λ=0.1) total={tot_c!r} ≠ HEAD → λ항이 실제로 손실에 반영됨 ✅")
    sys.exit(0 if same and tot_a != tot_c else 1)
