import shutil
import numpy as np
import torch
import torch.nn as nn
import time
import math
from pathlib import Path
import copy

from collections import defaultdict
from utils.data.load_data import create_data_loaders
from utils.common.utils import save_reconstructions, ssim_loss
from utils.common.loss_function import SSIMLoss
from utils.model.varnet import VarNet

import os

def train_epoch(args, epoch, model, data_loader, optimizer, loss_type, scheduler=None):
    model.train()
    start_epoch = start_iter = time.perf_counter()
    len_loader = len(data_loader)
    total_loss = 0.

    for iter, data in enumerate(data_loader):
        if getattr(args, "max_iters", 0) and iter >= args.max_iters:
            break
        mask, kspace, target, maximum, _, _ = data
        mask = mask.cuda(non_blocking=True)
        kspace = kspace.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        maximum = maximum.cuda(non_blocking=True)

        output = model(kspace, mask)
        loss = loss_type(output, target, maximum)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()   # per-iter (warmup→cosine)
        total_loss += loss.item()

        if iter % args.report_interval == 0:
            print(
                f'Epoch = [{epoch:3d}/{args.num_epochs:3d}] '
                f'Iter = [{iter:4d}/{len(data_loader):4d}] '
                f'Loss = {loss.item():.4g} '
                f'Time = {time.perf_counter() - start_iter:.4f}s',
            )
            start_iter = time.perf_counter()
    total_loss = total_loss / len_loader
    return total_loss, time.perf_counter() - start_epoch


def validate(args, model, data_loader):
    model.eval()
    reconstructions = defaultdict(dict)
    targets = defaultdict(dict)
    start = time.perf_counter()

    with torch.no_grad():
        for iter, data in enumerate(data_loader):
            mask, kspace, target, _, fnames, slices = data
            kspace = kspace.cuda(non_blocking=True)
            mask = mask.cuda(non_blocking=True)
            output = model(kspace, mask)

            for i in range(output.shape[0]):
                reconstructions[fnames[i]][int(slices[i])] = output[i].cpu().numpy()
                targets[fnames[i]][int(slices[i])] = target[i].numpy()

    for fname in reconstructions:
        reconstructions[fname] = np.stack(
            [out for _, out in sorted(reconstructions[fname].items())]
        )
    for fname in targets:
        targets[fname] = np.stack(
            [out for _, out in sorted(targets[fname].items())]
        )
    metric_loss = sum([ssim_loss(targets[fname], reconstructions[fname]) for fname in reconstructions])
    num_subjects = len(reconstructions)
    return metric_loss, num_subjects, reconstructions, targets, None, time.perf_counter() - start


def steps_per_epoch(args, len_loader):
    """--max-iters면 epoch당 실제 train step은 min(max_iters, len_loader)."""
    mi = getattr(args, "max_iters", 0)
    return min(mi, len_loader) if mi and mi > 0 else len_loader


def build_scheduler(optimizer, args, len_loader):
    """SDUM & warmup_iters>0이면 linear warmup→cosine LambdaLR, 아니면 None(상수 LR)."""
    if getattr(args, "model", "varnet") != "sdum" or getattr(args, "warmup_iters", 0) <= 0:
        return None
    total_iters = args.num_epochs * len_loader
    warmup = args.warmup_iters

    def lr_lambda(step):
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, total_iters - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def check_resume_compat(saved_args, args):
    """resume 시 저장 config와 현재 config가 일치해야 함(아키텍처·cosine horizon)."""
    for f in ("model", "num_cascades", "channels", "num_epochs", "warmup_iters", "lr"):
        s, c = getattr(saved_args, f, None), getattr(args, f, None)
        if s != c:
            raise ValueError(
                f"--resume 불일치: '{f}' 저장={s} 현재={c}. 같은 config로 재개해야 함(원래 계획까지만)."
            )


def load_resume_checkpoint(exp_dir, args):
    """model.pt(마지막 epoch) 로드 + config 일치 확인. 없으면 FileNotFoundError."""
    ckpt_path = Path(exp_dir) / "model.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"--resume인데 checkpoint 없음: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    check_resume_compat(ckpt["args"], args)
    return ckpt


def load_val_log(file_path):
    """val_loss_log 이어쓰기용. np.save는 '.npy'를 붙이므로 그 경로를 확인."""
    p = file_path + ".npy"
    return np.load(p) if os.path.exists(p) else np.empty((0, 2))


def save_model(args, exp_dir, epoch, model, optimizer, best_val_loss, is_new_best):
    torch.save(
        {
            'epoch': epoch,
            'args': args,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_val_loss': best_val_loss,
            'exp_dir': exp_dir
        },
        f=exp_dir / 'model.pt'
    )
    if is_new_best:
        shutil.copyfile(exp_dir / 'model.pt', exp_dir / 'best_model.pt')

        
def train(args):
    device = torch.device(f'cuda:{args.GPU_NUM}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)
    print('Current cuda device: ', torch.cuda.current_device())

    if getattr(args, "model", "varnet") == "sdum":
        from utils.model.sdum.config import tiny_8gb_args
        from utils.model.sdum.adapter import build_sdum
        sdum_args = tiny_8gb_args(num_cascades=args.num_cascades, channels=list(args.channels))
        model = build_sdum(sdum_args)
    else:
        model = VarNet(num_cascades=args.cascade,
                       chans=args.chans,
                       sens_chans=args.sens_chans)
    model.to(device=device)

    resume_ckpt = None
    if getattr(args, "resume", False):
        resume_ckpt = load_resume_checkpoint(args.exp_dir, args)
        model.load_state_dict(resume_ckpt["model"])

    loss_type = SSIMLoss().to(device=device)
    if getattr(args, "model", "varnet") == "sdum":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    else:
        optimizer = torch.optim.Adam(model.parameters(), args.lr)

    best_val_loss = 1.
    start_epoch = 0
    if resume_ckpt is not None:
        start_epoch = resume_ckpt["epoch"]
        best_val_loss = resume_ckpt["best_val_loss"]
        if torch.is_tensor(best_val_loss):
            best_val_loss = best_val_loss.item()   # best_val_loss를 python float로 정규화(fresh run의 1.0과 타입 일치; is_new_best 비교 안전)
        print(f"[resume] {args.exp_dir/'model.pt'} → start_epoch={start_epoch} best_val={best_val_loss:.4f}")


    train_loader = create_data_loaders(data_path = args.data_path_train, args = args, shuffle=True)
    val_loader = create_data_loaders(data_path = args.data_path_val, args = args)

    scheduler = build_scheduler(optimizer, args, len(train_loader))
    if scheduler is not None:
        print(f"[scheduler] warmup={args.warmup_iters} → cosine, "
              f"total_iters={args.num_epochs * len(train_loader)}, base_lr={args.lr}")

    if resume_ckpt is not None:
        optimizer.load_state_dict(resume_ckpt["optimizer"])   # scheduler 생성 후에 로드(initial_lr=base 유지)
        if scheduler is not None:
            for _ in range(start_epoch * steps_per_epoch(args, len(train_loader))):
                scheduler.step()
            print(f"[resume] scheduler fast-forward {start_epoch}ep, lr={optimizer.param_groups[0]['lr']:.3e}")

    run_tag = getattr(args, "run_tag", "")
    fname = f"val_loss_log_{run_tag}" if run_tag else "val_loss_log"
    val_log_path = os.path.join(args.val_loss_dir, fname)
    val_loss_log = load_val_log(val_log_path) if resume_ckpt is not None else np.empty((0, 2))
    for epoch in range(start_epoch, args.num_epochs):
        print(f'Epoch #{epoch:2d} ............... {args.net_name} ...............')
        
        train_loss, train_time = train_epoch(args, epoch, model, train_loader, optimizer, loss_type, scheduler)
        val_loss, num_subjects, reconstructions, targets, inputs, val_time = validate(args, model, val_loader)
        
        val_loss = val_loss / num_subjects   # §5-A: per-subject 평균으로 정규화(저장 전). 이전엔 미정규화 SUM을 저장했음
        val_loss_log = np.append(val_loss_log, np.array([[epoch, val_loss]]), axis=0)
        # run_tag/fname/val_log_path: §5-B(run별 분리 저장) — resume 시 이어쓰기와 같은 경로 재사용(위에서 계산)
        np.save(val_log_path, val_loss_log)
        print(f"loss file saved! {val_log_path}")

        train_loss = torch.tensor(train_loss).cuda(non_blocking=True)
        val_loss = torch.tensor(val_loss).cuda(non_blocking=True)
        is_new_best = val_loss < best_val_loss
        best_val_loss = min(best_val_loss, val_loss)

        save_model(args, args.exp_dir, epoch + 1, model, optimizer, best_val_loss, is_new_best)
        print(
            f'Epoch = [{epoch:4d}/{args.num_epochs:4d}] TrainLoss = {train_loss:.4g} '
            f'ValLoss = {val_loss:.4g} TrainTime = {train_time:.4f}s ValTime = {val_time:.4f}s',
        )

        if is_new_best:
            print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@NewRecord@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
            start = time.perf_counter()
            save_reconstructions(reconstructions, args.val_dir, targets=targets, inputs=inputs)
            print(
                f'ForwardTime = {time.perf_counter() - start:.4f}s',
            )
