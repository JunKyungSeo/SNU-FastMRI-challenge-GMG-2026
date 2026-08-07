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
from utils.common.loss_function import SSIMLoss, bbox_ssim_loss, masked_l1_loss
from utils.model.varnet import VarNet

import os

EMA_DECAY = 0.999


class EMA:
    """가중치 지수이동평균(θ_EMA ← d·θ_EMA + (1−d)·θ). 학습은 원본 θ로 하고, 검증과
    best_model.pt의 'model' 키만 θ_EMA로 남긴다 — 채점·--init-from 쪽은 무변경.

    d = min(EMA_DECAY, (1+n)/(10+n)) — 초반 보정. 없으면 shadow가 수백 step 동안
    초기 가중치에 끌려 첫 epoch val이 원본보다 나빠 보인다.
    """

    def __init__(self, model, decay=EMA_DECAY):
        self.decay = decay
        self.updates = 0
        # ⚠️ state_dict()는 named_parameters()와 달리 **중복 제거를 안 한다.** 한 텐서가 두 이름으로
        # 등록돼 있으면 m개 키로 나온다(프로덕션 config에서 1979키 중 431키가 중복):
        #   promptmr_plus.py:570 self.sens_net_ckp = ReentrantWrapper(self.sens_net) — 모듈 이중 등록
        #   promptmr_plus.py:277 act=nn.PReLU() mutable default — 스칼라 1개가 306 경로에 공유
        # 그래서 shadow도 **같은 텐서를 가리키게** 만들고(_keys = 대표 키만) update·swap이 대표만 돈다.
        # 이걸 안 하면 swap이 별칭을 m번 교환해 자기 역원이 아니게 되고(m=2면 위수 3 순열:
        # (θ, s₁, s₂) → (s₂, θ, s₁)), 검증 뒤 학습이 θ가 아니라 **θ_EMA에서 재개**된다.
        # 키 집합 자체는 그대로 저장되므로 기존 체크포인트·resume 호환이 안 변한다.
        self.shadow = {}
        self._keys = []
        rep_of = {}
        for k, v in model.state_dict().items():
            ptr = v.data_ptr() if v.numel() else None   # 빈 텐서는 data_ptr이 0으로 겹칠 수 있다
            rep = rep_of.get(ptr) if ptr is not None else None
            if rep is None:
                if ptr is not None:
                    rep_of[ptr] = k
                self._keys.append(k)
                self.shadow[k] = v.detach().clone()
            else:
                self.shadow[k] = self.shadow[rep]   # 별칭은 대표와 같은 객체를 공유

    @torch.no_grad()
    def update(self, model):
        self.updates += 1
        d = min(self.decay, (1 + self.updates) / (10 + self.updates))
        sd = model.state_dict()
        for k in self._keys:
            v = sd[k]
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(d).add_(v, alpha=1 - d)
            else:
                self.shadow[k].copy_(v)   # int buffer(카운터 등)는 평균이 무의미 → 최신값

    @torch.no_grad()
    def swap(self, model):
        """모델 가중치 ↔ shadow 제자리 교환(자기 역원). 검증 전후 한 번씩 — 백업 사본이
        따로 없어 상주 메모리는 shadow 한 벌뿐이다.
        대표 키만 도는 게 자기 역원의 조건이다 — 별칭까지 돌면 같은 텐서를 m번 교환한다."""
        sd = model.state_dict()
        for k in self._keys:
            v = sd[k]
            tmp = v.detach().clone()
            v.copy_(self.shadow[k])
            self.shadow[k].copy_(tmp)

    def state_dict(self):
        return {"updates": self.updates, "shadow": self.shadow}

    def load_state_dict(self, sd):
        self.updates = int(sd["updates"])
        for k in self._keys:                # 별칭은 대표와 같은 객체라 자동으로 따라온다
            self.shadow[k].copy_(sd["shadow"][k])   # copy_라 ckpt(cpu)→현재 device 이동이 자동


def train_epoch(args, epoch, model, data_loader, optimizer, loss_type, scheduler=None,
                skip_iters=0, best_val_loss=1., ema=None):
    """skip_iters: epoch 중간 재개 시 건너뛸 마이크로배치 수(latest.pt의 'iter').

    건너뛰기는 data_loader를 그냥 돌리는 방식이라 h5 읽기·SVD 비용은 그대로 든다. 대신 GPU
    forward/backward를 안 하므로 정상 학습보다는 빠르다. 순열 재현은 train()이 epoch마다
    generator를 seed+epoch로 다시 심어서 보장한다."""
    model.train()
    start_epoch = start_iter = time.perf_counter()
    total_loss = 0.
    fg = getattr(args, "fg_loss", False)   # B5: 켜지면 tuple에 fg_mask가 하나 더 붙고 손실을 그 안에서만 계산
    lam = float(getattr(args, "bbox_loss_lambda", 0.0))   # C1: >0이면 tuple 맨 끝에 bbox가 붙고 λ항을 더함
    use_bbox = lam > 0
    area_weight = bool(getattr(args, "area_weight", False))   # PromptMR+: masked SSIM × sqrt(면적비)
    l1_w = float(getattr(args, "l1_weight", 0.0))             # C11: >0이면 (1-w)·SSIM + w·L1
    accum = max(1, int(getattr(args, "accum_steps", 1)))      # gradient accumulation
    grad_clip = float(getattr(args, "grad_clip", 0.0))        # gradient clipping max_norm (0=off)
    gnorms = []   # 자르기 전 총 grad norm — clip 값이 실제로 발동하는지 보려고 모은다

    def _do_step():
        if grad_clip > 0:
            gnorms.append(float(torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)))
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)   # 0으로 채우지 말고 버퍼째 해제(그만큼 VRAM 반환)
        if scheduler is not None:
            scheduler.step()   # optimizer-step당 1회 (warmup→cosine)
        if ema is not None:
            ema.update(model)   # optimizer-step 경계에서만 — 누적 중간 가중치는 안 섞인다

    optimizer.zero_grad(set_to_none=True)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()   # epoch 단위 peak 측정(아래 로그에서 출력)
    pending = 0     # backward는 됐지만 아직 optimizer.step 안 한 마이크로배치 수
    processed = 0
    last_saved = skip_iters     # 마지막으로 latest.pt에 남긴 iter 위치
    if skip_iters > 0:
        print(f'[resume] epoch {epoch}: iter 0~{skip_iters - 1} 건너뛰는 중 '
              f'(가중치는 iter {skip_iters} 시점 것) ...')
        skip_start = time.perf_counter()

    for iter, data in enumerate(data_loader):
        if getattr(args, "max_iters", 0) and iter >= args.max_iters:
            break
        if iter < skip_iters:
            # 저장 지점까지 데이터만 흘려보낸다(모델·optimizer는 이미 그 시점 상태로 복원됨).
            if (iter + 1) % args.report_interval == 0:
                print(f'[resume] skip {iter + 1}/{skip_iters} '
                      f'({time.perf_counter() - skip_start:.0f}s)')
            continue
        if iter == skip_iters and skip_iters > 0:
            print(f'[resume] 건너뛰기 완료 ({time.perf_counter() - skip_start:.0f}s) '
                  f'→ iter {skip_iters}부터 학습 재개')
            start_iter = time.perf_counter()
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
        if l1_w > 0:
            # 볼록 결합이라 총 스케일이 기존과 같게 유지된다(항을 더하는 bbox 방식과 다른 점).
            # area_weight보다 **먼저** 섞어 두 항이 같은 면적 가중을 받게 한다.
            loss = (1 - l1_w) * loss + l1_w * masked_l1_loss(
                output, target, maximum, mask=(fg_mask if fg else None))
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

        # optimizer step 경계(pending==0)에서만 저장한다. 누적 도중에 저장하면 아직 step 안 된
        # 마이크로배치들의 gradient가 재개 시 사라진다. '>= ckpt_every'라서 accum과 ckpt_every가
        # 서로 안 나누어떨어져도 첫 step 경계에서 정확히 한 번 걸린다.
        if CKPT_EVERY_ITERS > 0 and pending == 0 and (iter + 1) - last_saved >= CKPT_EVERY_ITERS:
            save_latest(args, args.exp_dir, epoch, iter + 1, model, optimizer, best_val_loss, ema=ema)
            last_saved = iter + 1

        if iter % args.report_interval == 0:
            print(
                f'Epoch = [{epoch:3d}/{args.num_epochs:3d}] '
                f'Iter = [{iter:4d}/{len(data_loader):4d}] '
                f'Loss = {loss.item():.4g} '
                + (f'|g| = {gnorms[-1]:.3g} ' if gnorms else '')
                + f'Time = {time.perf_counter() - start_iter:.4f}s',
            )
            start_iter = time.perf_counter()

    if pending > 0:      # 꼬리: 남은 누적분 step (epoch 끝/max-iters 경계)
        _do_step()
    total_loss = total_loss / max(1, processed)
    if gnorms:
        g = sorted(gnorms)
        over = sum(1 for x in g if x > grad_clip)
        print(f'[gnorm] epoch {epoch} n={len(g)} med={g[len(g) // 2]:.3g} '
              f'p90={g[int(len(g) * 0.9)]:.3g} max={g[-1]:.3g} '
              f'clip({grad_clip}) 발동 {100 * over / len(g):.1f}%')
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
    """promptmr에서 warmup 또는 cosine 감쇠가 켜져 있으면 LambdaLR, 아니면 None(상수 LR).

    곡선은 세 구간이다 — [0, warmup) linear warmup → [warmup, decay_start) **base_lr 유지**
    → [decay_start, total) cosine 감쇠(하한 = base_lr × --lr-min-factor).
    warmup_iters·total_iters·decay_start는 전부 optimizer-step 단위(accum_steps로 나눔)다.
    --lr-decay-start-epoch < 0 이면 decay_start = warmup → 기존 동작(warmup 직후부터 전 구간 cosine).
    """
    if getattr(args, "model", "varnet") != "promptmr":
        return None
    warmup = int(getattr(args, "warmup_iters", 0))
    decay_start_epoch = int(getattr(args, "lr_decay_start_epoch", -1))
    if warmup <= 0 and decay_start_epoch < 0:
        return None
    accum = max(1, int(getattr(args, "accum_steps", 1)))
    # epoch당 optimizer step = ceil(len/accum) (train_epoch 꼬리 step 포함). floor면 스케줄이 어긋남.
    # ⚠️ len_loader가 아니라 steps_per_epoch()를 쓴다 — train_epoch은 --max-iters에서 break 하므로
    #    실제로 밟는 step이 그만큼 적다. len_loader를 쓰면 horizon이 길어져 cosine이 중간에서 끝나고,
    #    resume fast-forward(아래 train())는 steps_per_epoch()를 써서 눈금까지 어긋난다.
    opt_steps_per_epoch = math.ceil(steps_per_epoch(args, len_loader) / accum)
    total_iters = args.num_epochs * opt_steps_per_epoch   # optimizer-step 수
    decay_start = decay_start_epoch * opt_steps_per_epoch if decay_start_epoch >= 0 else warmup
    f_min = float(getattr(args, "lr_min_factor", 0.0))

    def lr_lambda(step):
        if step < warmup:
            return (step + 1) / warmup
        if step < decay_start:
            return 1.0
        progress = (step - decay_start) / max(1, total_iters - decay_start)
        # min(progress, 1.0) 필수 — horizon을 넘겨 밟으면 cos가 다시 올라가 LR이 역주행한다.
        return f_min + (1.0 - f_min) * 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def should_early_stop(val_loss_log, patience):
    """val이 patience epoch 연속 개선되지 않았으면 True (patience<=0이면 항상 False).

    카운터를 따로 들고 다니지 않고 val_loss_log((n,2) = [epoch, val_loss])에서 **파생**한다 —
    체크포인트에 새 필드를 넣을 필요가 없고, resume 시 load_val_log가 복원해 주므로 공짜로 이어진다.
    """
    if patience <= 0 or len(val_loss_log) == 0:
        return False
    return len(val_loss_log) - 1 - int(np.argmin(val_loss_log[:, 1])) >= patience


def check_resume_compat(saved_args, args):
    """resume 시 저장 config와 현재 config가 일치해야 함(아키텍처·cosine horizon).
    num_epochs·max_iters는 cosine horizon(total_iters=num_epochs*ceil(steps_per_epoch/accum))에만
    영향 → cosine 활성(warmup_iters>0 또는 lr_decay_start_epoch>=0)일 때만 일치를 요구한다.
    상수 LR에선 num_epochs 증가(=학습 연장)를 허용:
    루프 range(start_epoch, num_epochs)만 늘어날 뿐 LR 곡선이 바뀌지 않아 안전."""
    # 신규 필드가 없는 옛 체크포인트는 "기능 부재 시의 기본값"으로 간주(현재 값이 아님 — 그러면
    # 압축 없이 학습한 ckpt를 압축 켜고 재개하는 걸 잘못 허용). 압축 off(False)/nvc 8이 부재 기본.
    ABSENT_DEFAULT = {"coil_compress": False, "n_virtual_coils": 8,
                      "k_fold": 0, "val_fold": 0, "kfold_seed": 430,
                      "remask": "off", "remask_offset": "uniform", "l1_weight": 0.0,
                      "lr_decay_start_epoch": -1, "lr_min_factor": 0.0,
                      "mask_aug": "off", "early_stop_patience": 0, "max_iters": 0,
                      "holdout_tail": False}
    fields = ["model", "num_cascades", "n_feat0", "sens_n_feat0", "n_history",
              "warmup_iters", "lr", "accum_steps", "coil_compress", "k_fold",
              "remask", "l1_weight", "lr_decay_start_epoch", "lr_min_factor", "mask_aug",
              "holdout_tail"]
    # early_stop_patience는 일부러 뺐다 — 학습 결과를 바꾸지 않으므로 재개하면서 조정할 수 있어야 쓸모가 있다.
    # ⚠️ 위 필드명은 argparse dest 와 **글자 그대로 같아야** 한다. 어긋나면 getattr 이 양쪽 다
    #    ABSENT_DEFAULT 를 집어 항상 일치 → 가드가 영구 no-op 이 된다(조용히 학습셋이 바뀐 채 재개).
    # K-Fold에서 분할이 달라지면 train/val 경계가 바뀐다 → 재개하는 순간 이전 val 볼륨으로 학습하게 된다
    # (조용히 새는 leakage라 로그만 봐선 못 잡는다). 분할에 영향을 주는 값은 k-fold일 때만 확인한다.
    if int(getattr(saved_args, "k_fold", 0)) > 0 or int(getattr(args, "k_fold", 0)) > 0:
        fields += ["val_fold", "kfold_seed"]
    if (int(getattr(args, "warmup_iters", 0)) > 0
            or int(getattr(args, "lr_decay_start_epoch", -1)) >= 0):   # cosine horizon 있을 때만 고정
        # max_iters도 horizon이다 — total_iters = num_epochs * ceil(steps_per_epoch(args,len)/accum)
        # 이라 --max-iters를 바꿔 재개하면 곡선과 fast-forward 눈금이 함께 어긋난다.
        fields += ["num_epochs", "max_iters"]
    # --holdout-tail이면 상수 LR(=위 조건 거짓)이라도 max_iters를 묶는다. 그 값이 곧 샘플러의 n이고
    # n은 HoldoutTailSampler가 섞는 순열의 **길이**라, 1만 달라져도 epoch 구성이 통째로 바뀐다
    # → 중간 재개가 다른 순열에서 skip_iters를 세어 hold-out을 두 번 보거나 아예 안 보게 된다.
    # (RandomSampler 시절엔 max_iters가 뒤를 자르기만 해서 무해했다.)
    elif getattr(saved_args, "holdout_tail", False) or getattr(args, "holdout_tail", False):
        fields += ["max_iters"]
    # n_virtual_coils는 압축이 켜져 있을 때만 입력에 영향을 준다. 압축 off끼리 재개하는데
    # 죽은 값이 다르다고 막으면 헛돈다(코일 전부 사용이 기본이 된 뒤로는 이 경우가 정상 경로다).
    if getattr(saved_args, "coil_compress", False) or getattr(args, "coil_compress", False):
        fields.append("n_virtual_coils")
    # remask_offset·seed 는 재생성이 켜져 있을 때만 입력을 바꾼다(off 끼리 재개는 막지 않는다).
    if (str(getattr(saved_args, "remask", "off")) != "off"
            or str(getattr(args, "remask", "off")) != "off"):
        fields += ["remask_offset", "seed"]
    # --mask-aug 도 마스크를 (seed, epoch, volume, slice)에서 뽑는다 → seed가 바뀌면 입력이 바뀐다.
    elif (str(getattr(saved_args, "mask_aug", "off")) != "off"
            or str(getattr(args, "mask_aug", "off")) != "off"):
        fields.append("seed")
    for f in fields:
        fallback = ABSENT_DEFAULT.get(f, None)
        s, c = getattr(saved_args, f, fallback), getattr(args, f, fallback)
        if s != c:
            raise ValueError(
                f"--resume 불일치: '{f}' 저장={s} 현재={c}. 같은 config로 재개해야 함."
            )


def _ckpt_pos(path):
    """체크포인트의 진행 위치 (epoch, iter)만 확인하고 즉시 해제.

    두 후보를 동시에 들고 비교하면 345MB×2가 한꺼번에 떠서, 이 기능이 막으려는 바로 그 OOM을
    유발한다(노드 여유가 수백 MB뿐). 그래서 위치만 훑고 버린 뒤 이긴 쪽만 다시 로드한다 —
    로드 한 번(~2초)을 더 내고 동시 점유를 피하는 거래."""
    if not path.exists():
        return None
    ck = torch.load(path, map_location="cpu", weights_only=False)
    # model.pt의 'epoch'는 완료한 epoch 수 = 다음 epoch의 iter 0 지점. latest.pt는 진행 중 epoch + iter.
    pos = (int(ck["epoch"]), int(ck.get("iter", 0)))
    del ck
    return pos


def load_resume_checkpoint(exp_dir, args):
    """model.pt(epoch 경계) / latest.pt(epoch 중간) 중 **더 진행된** 쪽 로드 + config 일치 확인.

    (epoch, iter) 튜플 비교라 epoch가 정상 완료돼 model.pt가 (E+1, 0)이 되면 그게 자동으로
    같은 epoch의 latest.pt (E, 4500)를 이긴다 → 낡은 중간 체크포인트를 지울 필요가 없다."""
    exp_dir = Path(exp_dir)
    cands = {n: _ckpt_pos(exp_dir / n) for n in ("model.pt", "latest.pt")}
    cands = {n: p for n, p in cands.items() if p is not None}
    if not cands:
        raise FileNotFoundError(f"--resume인데 checkpoint 없음: {exp_dir}/(model.pt|latest.pt)")
    name = max(cands, key=cands.get)
    ckpt = torch.load(exp_dir / name, map_location="cpu", weights_only=False)
    check_resume_compat(ckpt["args"], args)
    print(f"[resume] 후보 {dict(cands)} → {name} 선택")
    return ckpt


def load_val_log(file_path):
    """val_loss_log 이어쓰기용. np.save는 '.npy'를 붙이므로 그 경로를 확인."""
    p = file_path + ".npy"
    return np.load(p) if os.path.exists(p) else np.empty((0, 2))


def save_model(args, exp_dir, epoch, model, optimizer, best_val_loss, is_new_best, ema=None):
    ckpt = {
        'epoch': epoch,
        'args': args,
        'model': model.state_dict(),
        'ema': ema.state_dict() if ema is not None else None,   # resume용 — 'model'은 원본 θ 유지
        'optimizer': optimizer.state_dict(),
        'best_val_loss': best_val_loss,
        'exp_dir': exp_dir
    }
    torch.save(ckpt, f=exp_dir / 'model.pt')
    if is_new_best:
        if ema is not None:
            # 채점(test_part)·--init-from은 best_model.pt의 'model'을 읽는다 → 검증에 쓴 θ_EMA를
            # 그 자리에 넣는다. 'ema'는 이때 'model'과 동일이라 뺀다(~파라미터×4B 디스크 절약).
            # resume은 model.pt/latest.pt만 보므로 best의 'ema' 부재는 안전하다.
            ckpt['model'] = ema.shadow
            ckpt['ema'] = None
            torch.save(ckpt, f=exp_dir / 'best_model.pt')
        else:
            shutil.copyfile(exp_dir / 'model.pt', exp_dir / 'best_model.pt')


# epoch 단위 저장(model.pt)만으로는 컨테이너가 죽을 때 그 epoch 진행분을 통째로 잃는다
# (2026-07-20 OOM 2건에서 실측 1시간 28분 유실). 그래서 epoch 중간에도 latest.pt를 남긴다.
CKPT_EVERY_ITERS = 500     # 0이면 iter 단위 저장 off


def save_latest(args, exp_dir, epoch, iter_done, model, optimizer, best_val_loss, ema=None):
    """epoch 중간 체크포인트를 latest.pt에 원자적으로 저장.

    model.pt와 별도 파일이다 — 저장 도중 죽어도 epoch 단위 체크포인트는 멀쩡해야 한다.
    tmp에 쓴 뒤 os.replace로 갈아끼운다(같은 파일시스템 내 rename은 원자적) → 저장 중 SIGKILL을
    맞아도 latest.pt는 항상 '직전에 완결된 저장분'이지 반쪽짜리가 아니다.

    Args:
        iter_done: 이 epoch에서 **완료한** 마이크로배치 수. 재개는 여기서부터 시작한다.
                   optimizer step 경계(pending==0)에서만 호출해야 누적분 유실이 없다.
    """
    tmp = exp_dir / 'latest.pt.tmp'
    torch.save(
        {
            'epoch': epoch,          # 진행 중인 epoch(완료 아님). model.pt의 epoch는 '완료+1'이라 의미가 다름.
            'iter': iter_done,
            'args': args,
            'model': model.state_dict(),
            'ema': ema.state_dict() if ema is not None else None,
            'optimizer': optimizer.state_dict(),
            'best_val_loss': best_val_loss,
            'exp_dir': exp_dir,
        },
        f=tmp,
    )
    os.replace(tmp, exp_dir / 'latest.pt')

        
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
    elif getattr(args, "init_from", ""):
        # Option C 미세조정: 공유 사전학습 가중치만 초기값으로 로드(optimizer/epoch/best_val은 새로 시작).
        # resume과 배타적 — resume이면 이 분기는 건너뛴다(위 if).
        init_ckpt = torch.load(args.init_from, map_location='cpu', weights_only=False)
        model.load_state_dict(init_ckpt["model"])
        print(f"[init-from] '{args.init_from}' 의 model 가중치로 초기화 (미세조정 시작)")

    # EMA는 항상 켠다(끄는 레버 없음). resume이면 저장된 shadow 복원, EMA 이전의 옛 ckpt면
    # 현재 가중치에서 새로 시작(그 시점부터의 평균 — 정확하진 않지만 이어갈 수 있는 최선).
    ema = EMA(model)
    if resume_ckpt is not None and resume_ckpt.get("ema") is not None:
        ema.load_state_dict(resume_ckpt["ema"])
        print(f"[resume] EMA shadow 복원 (updates={ema.updates})")

    loss_type = SSIMLoss().to(device=device)
    if getattr(args, "model", "varnet") == "promptmr":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    else:
        optimizer = torch.optim.Adam(model.parameters(), args.lr)

    best_val_loss = 1.
    start_epoch = 0
    start_iter = 0     # epoch 중간 재개 지점(latest.pt에서만 0이 아님)
    if resume_ckpt is not None:
        start_epoch = resume_ckpt["epoch"]
        start_iter = int(resume_ckpt.get("iter", 0))
        best_val_loss = resume_ckpt["best_val_loss"]
        if torch.is_tensor(best_val_loss):
            best_val_loss = best_val_loss.item()   # best_val_loss를 python float로 정규화(fresh run의 1.0과 타입 일치; is_new_best 비교 안전)
        print(f"[resume] start_epoch={start_epoch} start_iter={start_iter} best_val={best_val_loss:.4f}")


    if getattr(args, "k_fold", 0) > 0:
        from utils.data.load_data import create_kfold_data_loaders
        train_loader, val_loader = create_kfold_data_loaders(args)
    else:
        train_loader = create_data_loaders(data_path = args.data_path_train, args = args, shuffle=True)
        val_loader = create_data_loaders(data_path = args.data_path_val, args = args)

    scheduler = build_scheduler(optimizer, args, len(train_loader))
    if scheduler is not None:
        # build_scheduler와 같은 식으로 찍는다 — accum·--max-iters를 안 반영하면 로그가
        # 실제 horizon과 달라져, 스케줄이 의도대로 걸렸는지 확인할 유일한 계기판이 거짓말을 한다.
        _acc = max(1, int(getattr(args, "accum_steps", 1)))
        _spe = math.ceil(steps_per_epoch(args, len(train_loader)) / _acc)
        _de = int(getattr(args, "lr_decay_start_epoch", -1))
        print(f"[scheduler] warmup={args.warmup_iters} → "
              f"decay_start={_de * _spe if _de >= 0 else args.warmup_iters} step 부터 cosine "
              f"(하한 {float(getattr(args, 'lr_min_factor', 0.0))}×base), "
              f"total_iters={args.num_epochs * _spe}, base_lr={args.lr}")

    if resume_ckpt is not None:
        optimizer.load_state_dict(resume_ckpt["optimizer"])   # scheduler 생성 후에 로드(initial_lr=base 유지)
        if scheduler is not None:
            accum = max(1, int(getattr(args, "accum_steps", 1)))
            opt_steps_per_epoch = math.ceil(steps_per_epoch(args, len(train_loader)) / accum)   # ceil = 꼬리 step 포함(build_scheduler와 일치)
            # epoch 중간 재개면 그 epoch에서 이미 밟은 step(start_iter // accum)도 더해야 LR이 맞는다.
            # save_latest는 step 경계에서만 호출하므로 start_iter는 항상 accum의 배수라 나머지가 없다.
            for _ in range(start_epoch * opt_steps_per_epoch + start_iter // accum):
                scheduler.step()
            print(f"[resume] scheduler fast-forward {start_epoch}ep+{start_iter}it, "
                  f"lr={optimizer.param_groups[0]['lr']:.3e}")

    run_tag = getattr(args, "run_tag", "")
    fname = f"val_loss_log_{run_tag}" if run_tag else "val_loss_log"
    val_log_path = os.path.join(args.val_loss_dir, fname)
    val_loss_log = load_val_log(val_log_path) if resume_ckpt is not None else np.empty((0, 2))
    for epoch in range(start_epoch, args.num_epochs):
        print(f'Epoch #{epoch:2d} ............... {args.net_name} ...............')

        # 증강 스케줄에 epoch 전달(implementation.md §6b). val 로더는 is_train=False라 augmentor가 없다.
        # persistent_workers를 안 쓰므로 epoch마다 워커가 새로 떠 이 값이 워커 복사본에 실린다.
        tf = getattr(train_loader.dataset, "transform", None)
        if tf is not None:
            tf.set_epoch(epoch)
        augmentor = getattr(tf, "augmentor", None) if tf is not None else None
        if augmentor is not None:
            p = augmentor.prob()
            p_aff = p if augmentor.cfg.affine else 0.0
            print(f"[aug] epoch={epoch} p_flip={p:.4f} p_affine={p_aff:.4f}")
        elif getattr(args, "aug_on", False):
            raise RuntimeError("--aug-on 인데 augmentor가 배선되지 않았다 "
                               "(train_loader.dataset.transform.augmentor is None) — 증강 0회로 학습될 뻔했다.")

        # epoch 순열을 (seed, epoch)만의 함수로 고정 — 재개 시 같은 순서를 재현해야 저장된
        # iter까지 건너뛴 지점이 실제로 그때 그 지점이 된다. generator가 없으면(=val 로더 등)
        # 아무것도 안 한다.
        if getattr(train_loader, "generator", None) is not None:
            train_loader.generator.manual_seed(int(args.seed) + epoch)

        skip = start_iter if epoch == start_epoch else 0   # 중간 재개는 첫 epoch에만 적용
        train_loss, train_time = train_epoch(args, epoch, model, train_loader, optimizer, loss_type, scheduler,
                                             skip_iters=skip, best_val_loss=best_val_loss, ema=ema)
        ema.swap(model)   # 검증·best 판정·val 재구성 저장은 전부 θ_EMA로
        val_loss, num_subjects, reconstructions, targets, inputs, val_time = validate(args, model, val_loader)
        ema.swap(model)   # 원상복귀 — 다음 epoch 학습은 원본 θ로
        
        val_loss = val_loss / num_subjects   # §5-A: per-subject 평균으로 정규화(저장 전). 이전엔 미정규화 SUM을 저장했음
        val_loss_log = np.append(val_loss_log, np.array([[epoch, val_loss]]), axis=0)
        # run_tag/fname/val_log_path: §5-B(run별 분리 저장) — resume 시 이어쓰기와 같은 경로 재사용(위에서 계산)
        np.save(val_log_path, val_loss_log)
        print(f"loss file saved! {val_log_path}")

        train_loss = torch.tensor(train_loss).cuda(non_blocking=True)
        val_loss = torch.tensor(val_loss).cuda(non_blocking=True)
        is_new_best = val_loss < best_val_loss
        best_val_loss = min(best_val_loss, val_loss)

        save_model(args, args.exp_dir, epoch + 1, model, optimizer, best_val_loss, is_new_best, ema=ema)
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

        # early stopping — 저장이 다 끝난 뒤에 판정한다(best_model.pt는 위에서 이미 남았으므로 추가 저장 없음).
        patience = int(getattr(args, "early_stop_patience", 0))
        if should_early_stop(val_loss_log, patience):
            best_ep = int(val_loss_log[int(np.argmin(val_loss_log[:, 1])), 0])
            print(f"[early-stop] epoch {best_ep} 이후 {patience} epoch 연속 val 개선 없음 → 학습 중단 "
                  f"(best_val={float(val_loss_log[:, 1].min()):.4g})")
            break
