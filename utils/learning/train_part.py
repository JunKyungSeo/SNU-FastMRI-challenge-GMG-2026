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
from utils.common.loss_function import SSIMLoss, bbox_ssim_loss
from utils.model.varnet import VarNet

import os

def train_epoch(args, epoch, model, data_loader, optimizer, loss_type, scheduler=None):
    model.train()
    start_epoch = start_iter = time.perf_counter()
    total_loss = 0.
    fg = getattr(args, "fg_loss", False)   # B5: 켜지면 tuple에 fg_mask가 하나 더 붙고 손실을 그 안에서만 계산
    lam = float(getattr(args, "bbox_loss_lambda", 0.0))   # C1: >0이면 tuple 맨 끝에 bbox가 붙고 λ항을 더함
    use_bbox = lam > 0
    area_weight = bool(getattr(args, "area_weight", False))   # PromptMR+: masked SSIM × sqrt(면적비)
    accum = max(1, int(getattr(args, "accum_steps", 1)))      # gradient accumulation
    grad_clip = float(getattr(args, "grad_clip", 0.0))        # gradient clipping max_norm (0=off)

    def _do_step():
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)   # 0으로 채우지 말고 버퍼째 해제(그만큼 VRAM 반환)
        if scheduler is not None:
            scheduler.step()   # optimizer-step당 1회 (warmup→cosine)

    optimizer.zero_grad(set_to_none=True)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()   # epoch 단위 peak 측정(아래 로그에서 출력)
    pending = 0     # backward는 됐지만 아직 optimizer.step 안 한 마이크로배치 수
    processed = 0

    for iter, data in enumerate(data_loader):
        if getattr(args, "max_iters", 0) and iter >= args.max_iters:
            break
        # bbox는 규약상 항상 맨 끝(DataTransform docstring) → 먼저 떼어내면 나머지는 기존 언패킹 그대로.
        bboxes = None
        if use_bbox:
            bboxes = data[-1]
            data = data[:-1]
        if fg:
            mask, kspace, target, maximum, _, _ = data[:6]
            fg_mask = data[6].cuda(non_blocking=True)
        else:
            mask, kspace, target, maximum, _, _ = data
        mask = mask.cuda(non_blocking=True)
        kspace = kspace.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        maximum = maximum.cuda(non_blocking=True)

        output = model(kspace, mask)
        loss = loss_type(output, target, maximum, mask=fg_mask) if fg else loss_type(output, target, maximum)
        if area_weight and fg:
            # masked SSIM × sqrt(foreground 면적비). foreground 작은 슬라이스 손실을 낮춰 가중(작년 팀 방식).
            area_ratio = fg_mask.float().mean().clamp_min(1e-6)   # batch=1 → 스칼라
            loss = loss * area_ratio.sqrt()
        if use_bbox:
            # L = (1-SSIM_full) + λ·(1-SSIM_bbox). 박스 없는 슬라이스는 λ항이 정확히 0 → 기존 손실과 동일.
            loss = loss + lam * bbox_ssim_loss(loss_type, output, target, maximum, bboxes)

        (loss / accum).backward()   # accum 평균이 batch=accum 평균과 일치하도록 스케일
        pending += 1
        if pending == accum:
            _do_step()
            pending = 0
        total_loss += loss.item()   # 로깅은 raw loss
        processed += 1

        if iter % args.report_interval == 0:
            print(
                f'Epoch = [{epoch:3d}/{args.num_epochs:3d}] '
                f'Iter = [{iter:4d}/{len(data_loader):4d}] '
                f'Loss = {loss.item():.4g} '
                f'Time = {time.perf_counter() - start_iter:.4f}s',
            )
            start_iter = time.perf_counter()

    if pending > 0:      # 꼬리: 남은 누적분 step (epoch 끝/max-iters 경계)
        _do_step()
    total_loss = total_loss / max(1, processed)
    if torch.cuda.is_available():
        # 코일 압축을 뺀 뒤로는 peak 이 8GB에 얼마나 붙어 있는지가 상시 관심사 → epoch마다 남긴다.
        print(f'[mem] epoch {epoch} peak allocated = '
              f'{torch.cuda.max_memory_allocated() / 1024**3:.2f}GB / '
              f'reserved = {torch.cuda.max_memory_reserved() / 1024**3:.2f}GB')
    return total_loss, time.perf_counter() - start_epoch


def validate(args, model, data_loader):
    model.eval()
    reconstructions = defaultdict(dict)
    targets = defaultdict(dict)
    start = time.perf_counter()

    # B5: fg_loss면 검증도 채점 정합(masked SSIM_full)으로 계산 → val이 실채점 SSIM_full을 예측.
    fg = getattr(args, "fg_loss", False)
    if fg:
        fg_ssim = SSIMLoss().to(device=next(model.parameters()).device)
        fg_loss_sum, fg_count = 0.0, 0

    with torch.no_grad():
        for data in data_loader:
            if fg:
                mask, kspace, target, maximum, fnames, slices, fg_mask = data
            else:
                mask, kspace, target, _, fnames, slices = data
            kspace = kspace.cuda(non_blocking=True)
            mask = mask.cuda(non_blocking=True)
            output = model(kspace, mask)

            if fg:
                l = fg_ssim(output, target.cuda(non_blocking=True),
                            maximum.cuda(non_blocking=True), mask=fg_mask.cuda(non_blocking=True))
                fg_loss_sum += l.item() * output.shape[0]   # 슬라이스 단위 누적(batch=1이면 슬라이스=1)
                fg_count += output.shape[0]

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
    if fg:
        # train()이 val_loss/num_subjects로 나누므로 (합, 슬라이스수)를 넘겨 '슬라이스 평균 (1-SSIM_full)'이 되게 함.
        # → 채점의 슬라이스 평균 SSIM_full과 동일 단위. best-model 선택도 이 값(작을수록 좋음) 기준.
        return fg_loss_sum, max(1, fg_count), reconstructions, targets, None, time.perf_counter() - start

    metric_loss = sum([ssim_loss(targets[fname], reconstructions[fname]) for fname in reconstructions])
    num_subjects = len(reconstructions)
    return metric_loss, num_subjects, reconstructions, targets, None, time.perf_counter() - start


def steps_per_epoch(args, len_loader):
    """--max-iters면 epoch당 실제 train step은 min(max_iters, len_loader)."""
    mi = getattr(args, "max_iters", 0)
    return min(mi, len_loader) if mi and mi > 0 else len_loader


def build_scheduler(optimizer, args, len_loader):
    """promptmr & warmup_iters>0이면 linear warmup→cosine LambdaLR, 아니면 None(상수 LR).
    warmup_iters·total_iters는 optimizer-step 단위(accum_steps로 나눔)로 해석한다."""
    if getattr(args, "model", "varnet") != "promptmr" or getattr(args, "warmup_iters", 0) <= 0:
        return None
    accum = max(1, int(getattr(args, "accum_steps", 1)))
    # epoch당 optimizer step = ceil(len/accum) (train_epoch 꼬리 step 포함). floor면 스케줄이 어긋남.
    total_iters = args.num_epochs * math.ceil(len_loader / accum)   # optimizer-step 수
    warmup = args.warmup_iters

    def lr_lambda(step):
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, total_iters - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def check_resume_compat(saved_args, args):
    """resume 시 저장 config와 현재 config가 일치해야 함(아키텍처·cosine horizon).
    num_epochs는 cosine 스케줄 horizon(total_iters=num_epochs*len)에만 영향 → cosine 활성
    (warmup_iters>0)일 때만 일치를 요구한다. 상수 LR에선 num_epochs 증가(=학습 연장)를 허용:
    루프 range(start_epoch, num_epochs)만 늘어날 뿐 LR 곡선이 바뀌지 않아 안전."""
    # 신규 필드가 없는 옛 체크포인트는 "기능 부재 시의 기본값"으로 간주(현재 값이 아님 — 그러면
    # 압축 없이 학습한 ckpt를 압축 켜고 재개하는 걸 잘못 허용). 압축 off(False)/nvc 8이 부재 기본.
    ABSENT_DEFAULT = {"coil_compress": False, "n_virtual_coils": 8}
    fields = ["model", "num_cascades", "n_feat0", "sens_n_feat0", "n_history",
              "warmup_iters", "lr", "accum_steps", "coil_compress"]
    if int(getattr(args, "warmup_iters", 0)) > 0:   # cosine horizon 있을 때만 num_epochs 고정
        fields.append("num_epochs")
    # n_virtual_coils는 압축이 켜져 있을 때만 입력에 영향을 준다. 압축 off끼리 재개하는데
    # 죽은 값이 다르다고 막으면 헛돈다(코일 전부 사용이 기본이 된 뒤로는 이 경우가 정상 경로다).
    if getattr(saved_args, "coil_compress", False) or getattr(args, "coil_compress", False):
        fields.append("n_virtual_coils")
    for f in fields:
        fallback = ABSENT_DEFAULT.get(f, None)
        s, c = getattr(saved_args, f, fallback), getattr(args, f, fallback)
        if s != c:
            raise ValueError(
                f"--resume 불일치: '{f}' 저장={s} 현재={c}. 같은 config로 재개해야 함."
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

    if getattr(args, "model", "varnet") == "promptmr":
        from utils.model.promptmr.adapter import build_promptmr
        model = build_promptmr(args)
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
    if getattr(args, "model", "varnet") == "promptmr":
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
            accum = max(1, int(getattr(args, "accum_steps", 1)))
            opt_steps_per_epoch = math.ceil(steps_per_epoch(args, len(train_loader)) / accum)   # ceil = 꼬리 step 포함(build_scheduler와 일치)
            for _ in range(start_epoch * opt_steps_per_epoch):
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
