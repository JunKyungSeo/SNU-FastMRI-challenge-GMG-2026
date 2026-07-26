import json
import zlib
import numpy as np
import torch

from utils.common.metrics import foreground_mask  # B5: 채점과 동일한 foreground 마스크 정의 재사용(중복정의 금지)


def to_tensor(data):
    """
    Convert numpy array to PyTorch tensor. For complex arrays, the real and imaginary parts
    are stacked along the last dimension.
    Args:
        data (np.array): Input numpy array
    Returns:
        torch.Tensor: PyTorch version of data
    """
    return torch.from_numpy(data)


# ============================================================================
# Data Augmentation (strategy/augmentation.md 구현)
#   설계 원칙: hallucination-free lesion fidelity — bbox 안전 + 물리적으로 유효한 증강만.
#   채택(ON): 좌-우(width/phase-encode) 플립만 · ACS 보존 마스크 증강 · 소폭 정수 translation
#             · k-space 랜덤 위상(marginal) · 측정 노이즈(보수적)
#   위-아래(readout) 플립은 제외(무릎 뼈 위치가 뒤바뀌는 비현실 형태) — augmentation.md Tier 3.
#   스케줄: MRAugment schedule_p 형태 — delay D까지 off, T(=총 epoch)로 정규화한 exp/ramp/constant 증가.
#   epoch 전달(train_part.py set_epoch)·args 정의(train.py --aug-*)는 배선 완료.
#     --aug-on 기본값이 off라, 켜지 않으면 기존 학습과 동일하게 동작.
# ============================================================================

def _fftc(img, axes=(-2, -1)):
    """centered 2D FFT (fastmri 규약: ifftshift → fft2(ortho) → fftshift)."""
    img = np.fft.ifftshift(img, axes=axes)
    k = np.fft.fft2(img, axes=axes, norm="ortho")
    return np.fft.fftshift(k, axes=axes)


def _ifftc(k, axes=(-2, -1)):
    k = np.fft.ifftshift(k, axes=axes)
    img = np.fft.ifft2(k, axes=axes, norm="ortho")
    return np.fft.fftshift(img, axes=axes)


# ============================================================================
# 확정 파이프라인 전처리 (strategy/confirmed_model.md) — 아래 3개는 DataTransform이
# 플래그로 on/off. 기본 전부 off라, 켜기 전엔 기존 학습과 100% 동일하게 동작한다.
#   (1) readout crop  : k-space 높이(readout) 640→384만 crop, 폭(PE)은 native 유지
#   (2) 전역 스칼라 정규화 : 입력유도 스칼라 1개로 k-space·GRAPPA·target 동일 ÷s
#   (3) GRAPPA 정렬   : flip 증강 시 GRAPPA도 동기 (MRIAugmentor 내부)
# ============================================================================

def _ifftc1(x, axis):
    """1축(axis)만 centered iFFT (readout crop용). fastmri 규약(ortho)."""
    x = np.fft.ifftshift(x, axes=axis)
    x = np.fft.ifft(x, axis=axis, norm="ortho")
    return np.fft.fftshift(x, axes=axis)


def _fftc1(x, axis):
    """1축(axis)만 centered FFT."""
    x = np.fft.ifftshift(x, axes=axis)
    x = np.fft.fft(x, axis=axis, norm="ortho")
    return np.fft.fftshift(x, axes=axis)


def readout_crop(kspace_c, target_h=384):
    """readout(height=axis-2) oversampling 조기 제거. 폭(PE, axis-1)은 절대 안 건드림.
    height축만 이미지화 → center crop(640→target_h) → 다시 k-space.
    readout는 완전샘플이라 정확 가역 = 무손실(실측: utils/data/verify_readout_crop.py, corr 1.0).
    출력단 varnet center_crop(384)과 수학적으로 동일(이중 crop 아님, 조기 crop이면 이후 no-op).
    반환: (coils, target_h, W) complex — 입력과 같은 dtype으로 caller가 캐스팅할 것."""
    img_h = _ifftc1(kspace_c, axis=-2)          # (coils, H, W) — height만 이미지 도메인
    H = img_h.shape[-2]
    if H > target_h:
        st = (H - target_h) // 2
        img_h = img_h[..., st:st + target_h, :]
    elif H < target_h:                          # (드묾) 더 짧으면 center pad
        pad = target_h - H
        img_h = np.pad(img_h, [(0, 0)] * (img_h.ndim - 2) + [((pad + 1) // 2, pad // 2), (0, 0)])
    return _fftc1(np.ascontiguousarray(img_h), axis=-2)


def global_scale(kspace_c, percentile=99.0):
    """입력 유도 전역 스칼라 1개. zero-filled RSS 영상의 robust percentile(기본 p99).
    코일별 아님(전역) → 코일 상대비 보존 → sens map 추정 무해. acc4/acc8 무관 동일 규칙.
    ⚠️ label 통계(maximum attr) 대신 '입력'에서 뽑아야 학습·추론이 일관(추론엔 label 없음)."""
    rss = np.sqrt((np.abs(_ifftc(kspace_c)) ** 2).sum(0))   # (H, W) zero-filled RSS
    return float(np.percentile(rss, percentile)) + 1e-8


def _detect_acs(mask):
    """1D 마스크(1=sampled)에서 중앙 연속 완전샘플(ACS) 구간을 bool로 반환.
    ACS는 sensitivity map/GRAPPA 캘리브레이션 소스라 증강에서 절대 보존해야 한다."""
    W = mask.shape[0]
    c = W // 2
    if mask[c] < 1:  # 중앙이 안 찍혔으면 가장 가까운 샘플을 중심으로
        sampled = np.where(mask >= 1)[0]
        if sampled.size == 0:
            return np.zeros(W, dtype=bool)
        c = int(sampled[np.argmin(np.abs(sampled - c))])
    acs = np.zeros(W, dtype=bool)
    i = c
    while i >= 0 and mask[i] >= 1:
        acs[i] = True
        i -= 1
    i = c + 1
    while i < W and mask[i] >= 1:
        acs[i] = True
        i += 1
    return acs


def coil_compress(kspace_c, mask1d, n_virtual_coils=8):
    """SVD 기반 coil compression (SCC = Single-slice/SVD Coil Compression).

    ── 무엇을 하나 ────────────────────────────────────────────────────────────
    물리 코일 C개(예: 15)의 k-space를, 신호(SNR)가 큰 상위 `n_virtual_coils`개의
    **가상 코일(virtual coil)**로 선형 결합해 코일 차원을 줄인다.
    코일 수는 sens_net·모든 VarNet cascade의 연산·VRAM에 곱해지는 인자라,
    15→8이면 per-cascade 비용이 거의 코일 비율만큼(≈53%) 줄어든다.
    ⚠️ SSIM은 "상승"이 아니라 "거의 동등(neutral)"이 실측 결론(무압축 대비 PSNR 소폭↓,
       SSIM 동등: Benchmarking 3D multi-coil NC-PDNet, arXiv:2411.05883). 즉 **속도/VRAM
       레버**이지 화질 레버가 아니다. (근거·출처: strategy/preprocessing.md A2, related_work.md)

    ── 왜 ACS에서 압축행렬을 뽑나 (중요) ──────────────────────────────────────
    압축행렬 W는 "코일 조합 기저"라, 추론 시점에 **실제로 가진 데이터**로 계산해야 한다.
    대회 추론에서는 언더샘플 k-space만 주어지고, 그중 **완전샘플 구간은 중앙 ACS(저주파)뿐**이다.
    따라서 train/test 모두 ACS에서만 W를 구해 일관성을 유지한다(confirmed_model.md의
    "입력 유도(label 통계 금지)" 원칙과 동일한 계열). ACS는 mask 증강에도 100% 보존되므로
    (MRIAugmentor._aug_mask) 증강과도 안전하게 호환된다.

    ── 알고리즘 (SVD) ─────────────────────────────────────────────────────────
      1) ACS 열(중앙 완전샘플 PE 라인)만 모아 calibration 행렬 A = (C, H·n_acs) 구성.
      2) A = U Σ Vᴴ 로 SVD. 좌특이벡터 U(C×C)가 코일 간 상관을 에너지 순으로 담은 기저.
      3) 상위 Nvc열 W = U[:, :Nvc] 를 압축행렬로 사용.
      4) 가상 코일 v = Σ_c conj(W[c,v]) · kspace[c]  →  (Nvc, H, W).

    ── 파이프라인 안전성 (설계 근거) ──────────────────────────────────────────
    • 코일축 선형결합이라 spatial crop(readout_crop)·flip과 commute → 순서 무관, DataTransform은
      readout_crop 직후·normalize 직전에 호출한다(잘라 작아진 뒤 SVD → 더 쌈).
    • mask는 코일 무관(W,) → 불변. 미샘플 라인(값 0)은 0의 선형결합이라 그대로 0 유지 →
      VarNet의 DC(원 측정 k-space 기준)가 **압축 코일 공간에서 그대로 성립**한다.
    • GRAPPA 영상 prior는 이미 코일결합된 384² → 압축과 무관(영향 없음).
    • sens_net은 압축된 ACS로 내부 감도맵을 추정 → 그냥 Nvc코일로 동작(수정 불필요).

    ── 상호배타 주의 ─────────────────────────────────────────────────────────
    candidates.md 기준 coil compression은 **배치 SME(§3.2)와 택1(동시적용 금지)** — 둘 다
    SME 비용 절감이 목적이라 중복. 하나만 켠다.

    Args:
        kspace_c (np.ndarray): complex (C, H, W). readout_crop 이후라면 H=crop_h.
        mask1d   (np.ndarray): (W,) 0/1 언더샘플 마스크 (ACS 위치 판정용).
        n_virtual_coils (int): 남길 가상 코일 수(기본 8). C 이상이면 no-op.

    Returns:
        np.ndarray: complex (Nvc, H, W). C≤Nvc 또는 ACS 부재 시 입력을 그대로 반환(안전).
                    dtype은 numpy SVD가 complex128로 승격할 수 있으니, 호출부에서
                    orig_cdtype으로 캐스팅한다(DataTransform.__call__ 말미).
    """
    C = kspace_c.shape[0]
    if n_virtual_coils is None or n_virtual_coils >= C:
        return kspace_c                              # 압축 이득 없음 → no-op
    acs = _detect_acs(np.asarray(mask1d))            # bool (W,)
    if int(acs.sum()) < 1:
        return kspace_c                              # ACS 없으면 W 추정 불가 → 안전하게 원본
    # (1) calibration 행렬: ACS 열만 → (C, H*n_acs). 코일별 한 행.
    cal = kspace_c[:, :, acs].reshape(C, -1)
    # (2) SVD. full_matrices=False면 U:(C, min(C,N)). 좌특이벡터 = 코일 압축 기저.
    U, _, _ = np.linalg.svd(cal, full_matrices=False)
    # (3) 상위 Nvc열 = 압축행렬 W (C, Nvc).
    W = U[:, :n_virtual_coils]
    # (4) 적용: (Nvc, C) @ (C, H, W) → (Nvc, H, W). Wᴴ = conj(W).T.
    kc = np.tensordot(np.conj(W).T, kspace_c, axes=([1], [0]))
    return np.ascontiguousarray(kc)


class AugmentConfig:
    """증강 설정. train.py args에서 채워지며(없으면 getattr 기본값), 기본은 전부 off."""

    def __init__(self, **kw):
        self.on = kw.get("on", False)
        # 스케줄 — MRAugment 공식 repo(mraugment/data_augment.py: schedule_p)와 동일 형태.
        #   p(t<D)=0, p(T)=p_max. T로 정규화하므로 런 길이가 바뀌어도 증강량이 스케일된다.
        self.schedule = kw.get("schedule", "exp")          # 'exp' | 'ramp' | 'constant'
        self.delay = kw.get("delay", 0)                    # D: 증강 없이 보내는 앞쪽 epoch 수
        self.total_epochs = kw.get("total_epochs", 1)      # T: args.num_epochs (load_data가 채움)
        self.exp_decay = kw.get("exp_decay", 5.0)          # 논문/공식 repo 기본 c=5.0
        self.max_prob = kw.get("max_prob", 0.55)           # 논문 기본 p_max=0.55
        # 개별 aug on/off
        self.flip = kw.get("flip", True)                   # 좌-우(width)만
        self.mask = kw.get("mask", True)
        # 변환별 가중치(공식 repo --aug_weight_*): 실제 확률 = p · w_i. 한쪽만 0으로 두면 분리 A/B가 된다.
        self.weight_flip = kw.get("weight_flip", 1.0)
        self.weight_mask = kw.get("weight_mask", 1.0)
        self.phase = kw.get("phase", False)                # marginal → 기본 off
        self.noise = kw.get("noise", False)                # 검토 → 기본 off
        # ※ translation은 미채택: target이 384로 center-crop돼 제공되는데 모델은 640×368에서
        #   크롭하므로, 640 도메인 shift가 384 target과 정렬 불가(높이 크롭이 wrap과 불일치).
        #   flip은 center-crop과 commute해 안전하지만 translation은 안 됨. → 제거.
        # 파라미터
        self.mask_thin_max = kw.get("mask_thin_max", 4)    # 주변부 추가 언더샘플 최대 배수
        self.orig_mask_prob = kw.get("orig_mask_prob", 0.5)  # 원본(무증강) 마스크 유지 확률(분포 정합)
        self.noise_sigma = kw.get("noise_sigma", 0.0)      # 상대 측정 노이즈(보수적)
        self.seed = kw.get("seed", 430)


class MRIAugmentor:
    """augmentation.md의 데이터 증강을 numpy로 구현. DataTransform이 호출.
    모든 증강은 (kspace_complex, mask, target[, bbox])를 정렬 유지하며 함께 변환한다."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def probs(self):
        """(p, p_flip, p_mask). 학습 로그와 __call__이 같은 값을 쓰도록 한 곳에서 계산.
        p_flip·p_mask는 **실효 확률** — on/off 스위치(cfg.flip·cfg.mask)가 꺼져 있으면 0이다.
        (로그에 찍히는 숫자가 '실제로 이 증강이 걸릴 확률'과 일치해야 오해가 없다.)"""
        c = self.cfg
        p = self._prob()
        p_flip = min(1.0, max(0.0, p * float(c.weight_flip))) if c.flip else 0.0
        p_mask = (
            min(1.0, max(0.0, p * float(c.weight_mask) * (1.0 - float(c.orig_mask_prob))))
            if c.mask else 0.0
        )
        return p, p_flip, p_mask

    # ---- 스케줄 ----
    def _prob(self):
        """MRAugment 공식 schedule_p와 동일: D=delay, T=total_epochs, c=exp_decay/(T-D).
        p(t<D)=0이고 p(T)=p_max로 정규화되므로 런 길이가 달라도 증강량이 일관된다.
        (epoch은 0..T-1이라 마지막 epoch의 p는 p_max에 살짝 못 미친다 — 상류와 동일.)"""
        c = self.cfg
        if not c.on:
            return 0.0
        D, T, t, p_max = int(c.delay), int(c.total_epochs), self.epoch, float(c.max_prob)
        if t < D:
            return 0.0
        if c.schedule == "constant":
            return p_max
        if T - D <= 0:                       # 가드: 1 epoch 스모크 런 등
            return p_max
        if c.schedule == "ramp":
            return p_max * (t - D) / (T - D)
        k = c.exp_decay / (T - D)
        return float(p_max / (1.0 - np.exp(-(T - D) * k)) * (1.0 - np.exp(-(t - D) * k)))

    def _rng(self, fname, sl):
        """샘플별 결정적 RNG: (seed, epoch, fname, slice) → aug ON/OFF A/B가 동일 추출 위에서 비교됨.
        ⚠️ builtin hash()는 문자열 salt(PYTHONHASHSEED 미설정)로 프로세스마다 달라져 재현성이 깨진다
        (seed_fix로도 안 고쳐짐). 프로세스 안정 해시 zlib.crc32를 쓴다."""
        key = repr((self.cfg.seed, self.epoch, str(fname), int(sl))).encode()
        return np.random.default_rng(zlib.crc32(key))

    # ---- 개별 augmentation ----
    def _aug_mask(self, rng, mask):
        """ACS 보존 + 주변부(고주파) 추가 언더샘플. 원본 마스크의 부분집합만 반환(라인 추가 불가)."""
        acs = _detect_acs(mask)
        periph_idx = np.where((mask >= 1) & (~acs))[0]
        if periph_idx.size == 0:
            return mask.copy()
        thin = int(rng.integers(1, self.cfg.mask_thin_max + 1))  # 1..thin_max
        if thin == 1:
            return mask.copy()
        off = int(rng.integers(0, thin))
        keep = np.zeros_like(mask)
        keep[acs] = 1
        keep[periph_idx[off::thin]] = 1  # equispaced thinning + 랜덤 오프셋
        return keep.astype(mask.dtype)

    def _flip_lr(self, K, target, bbox, mask, grappa=None):
        """좌-우(width) 플립을 FFT 없이 closed form으로 수행 → 픽셀 정확 정렬(재시뮬 불필요).
        DFT 성질: y[n]=x[W-1-n]이면 Y[k]=e^{2πik/W}·X[-k]. 이 파일의 centered 규약
        (fftshift(fft(ifftshift(·))))에서 인덱스 -k는 centered 인덱스로 (W-j)%W에 대응하고,
        이는 마스크와 동일한 공식 roll(flip(·),1)이다. 즉 K2 = roll(flip(K),1) · ramp,
        ramp[j] = exp(2πi(j-W//2)/W). height축 FFT는 폭 flip에 대해 항등(취소)이라
        아예 필요 없고, width축 관계는 위 식으로 대수적으로 닫혀 있어 FFT 자체가 불필요하다
        (기존 ifftc→flip→fftc 왕복 대비 15배 빠름, 470ms→31ms @ (15,640,368) 실측).
        검증: 위 공식과 옛 이미지-도메인 왕복 구현의 상대오차 ~2.5e-07(W=368) — 짝수 폭 전제.
        홀수 W(예 371)는 공식이 깨진다(상대오차 ~1.4) → 아래 assert가 그 전제를 강제한다.
        마스크는 같은 even-length 정확 공식 M'[j]=M[(W-j)%W] = roll(flip(M),1)로 도출(1-라인 오차 방지).
        GRAPPA(영상 384², 있으면)도 같은 좌우 flip으로 동기 — 두 입력의 해부 정렬 유지."""
        assert mask.shape[0] % 2 == 0, (
            f"좌우 flip의 마스크 공식 roll(flip(M),1)은 짝수 폭에서만 성립한다 (W={mask.shape[0]}). "
            "홀수 폭이면 flip된 k-space 에너지가 마스크 밖으로 나가 조용히 삭제된다.")
        W = K.shape[-1]
        j = np.arange(W)
        ramp = np.exp(2j * np.pi * (j - W // 2) / W).astype(K.dtype, copy=False)
        K2 = np.roll(np.flip(K, axis=-1), 1, axis=-1) * ramp
        M2 = np.roll(np.flip(mask), 1).astype(np.float32)
        K2 = K2 * M2                                       # FFT 수치 누설 제거
        tgt = np.ascontiguousarray(target[:, ::-1])
        if bbox is not None:
            W = tgt.shape[1]
            bbox = [{**b, "x": W - int(b["x"]) - int(b["width"])} for b in bbox]
        g = None if grappa is None else np.ascontiguousarray(grappa[..., ::-1])
        return K2, tgt, M2, bbox, g

    def __call__(self, kspace_c, mask, target, bbox, grappa, fname, sl):
        """kspace_c:(coils,H,W) complex(raw) · mask:(W,) · target:(384,384) or None
        · grappa:(384,384) magnitude or None(GRAPPA 보조 입력, 확정: 영상 도메인 유지).
        반환: (kspace_masked_complex, mask_1d, target, bbox, grappa).
        ⚠️ GRAPPA에는 flip(기하)만 동기 적용하고 mask/phase/noise 증강은 미적용
           — 이미 복원된 깨끗한 prior라 라인 더 지우거나 노이즈 주입은 스스로 훼손(confirmed_model)."""
        cfg = self.cfg
        M = (mask >= 1).astype(mask.dtype)
        K = kspace_c * M                       # 제공 언더샘플 적용
        if (not cfg.on) or (target is None):
            return K, M, target, bbox, grappa
        p, p_flip, p_mask = self.probs()
        if p <= 0.0:
            return K, M, target, bbox, grappa
        rng = self._rng(fname, sl)
        tgt = target

        # 1) mask augmentation — 원본(고정)마스크 혼합을 p_mask에 이미 반영(test 분포 정합). GRAPPA엔 미적용.
        if cfg.mask and (rng.random() < p_mask):
            M = self._aug_mask(rng, M)
            K = kspace_c * M

        # 2) 좌-우(width) flip — target·bbox·mask·GRAPPA 동기(정렬 유지)
        if cfg.flip and (rng.random() < p_flip):
            K, tgt, M, bbox, grappa = self._flip_lr(K, tgt, bbox, M, grappa)

        # 3) k-space 랜덤 위상 (global, magnitude/RSS 불변 → bbox 채점 불변, 이득 marginal). GRAPPA 불변.
        if cfg.phase and (rng.random() < p):
            K = K * np.exp(1j * rng.uniform(0.0, 2.0 * np.pi))

        # 4) 측정 노이즈 (보수적, 샘플된 라인에만). GRAPPA 미적용.
        if cfg.noise and (cfg.noise_sigma > 0.0) and (rng.random() < p):
            scale = cfg.noise_sigma * float(np.abs(K).mean())
            n = (rng.standard_normal(K.shape) + 1j * rng.standard_normal(K.shape)) * scale
            K = K + n * M

        return K, M.astype(mask.dtype), np.ascontiguousarray(tgt), bbox, grappa


class DataTransform:
    """확정 파이프라인 전처리(strategy/confirmed_model.md). 신규 기능은 전부 플래그로 게이트.

    ── 다른 파일로의 전달 형식 (network/학습 코드 = 동료 담당) ──────────────────
    반환은 위치(positional) 튜플. **핵심 6개는 항상 고정** (기존과 동일):
        (mask, kspace, target, maximum, fname, slice)
    아래 플래그가 켜지면 그 뒤에 이 순서로만 덧붙는다 (bbox는 항상 맨 끝):
        normalize=True     → + scale        (float; 출력 역정규화용, output*scale)
        return_grappa=True → + grappa        (Tensor (384,384) real; regularizer prior 채널로 concat)
        return_fg_mask=True→ + fg_mask       (Tensor (384,384) 0/1; B5 손실·검증 정합용. forward엔 미포함)
        return_bbox=True   → + bbox          (list[dict]; 가변 길이라 _bbox_collate 필요)
    즉 최대 형태: (mask, kspace, target, maximum, fname, slice, [scale], [grappa], [fg_mask], [bbox]).

    ⚠️ 기본값 전부 off → 정확히 6-튜플 → train_part.py/test_part.py의 6개 언패킹과 비트 동일.
       플래그를 켜면 소비부(train_epoch/validate/test의 `... = data`)를 그에 맞게 확장해야 함.

    ── network 설계자에게 (플래그 켰을 때 받게 되는 것) ───────────────────────
      • kspace : readout_crop=True면 높이 640→crop_h(기본 384), 폭은 native(acc4 368/acc8 372).
                 normalize=True면 전역 스칼라로 나뉜 상태. shape=(coils, crop_h, W, 2) real/imag.
                 ⚠️ coil_compress=True면 coils=n_virtual_coils(예 8)로 축소된다(원 15가 아님).
                    sens_net·cascade는 그 코일 수로 그대로 동작(수정 불필요). DC도 압축공간에서 성립.
      • grappa : 영상 도메인 magnitude(384², fft 안 함). normalize 시 같은 s로 나뉨. flip 증강 시 동기.
                 → confirmed_model: 영상 prior 채널 concat + residual(out = grappa + f(...)).
                 ※ grappa 폭 384 vs kspace 폭 W(368/372) 격자 차 → network에서 폭 crop 또는 출력단 384² head로 정합.
      • scale  : per-sample 정규화 스칼라. 최종 출력·SSIM은 이 값으로 역정규화(원 스케일 복귀) 후 채점.
      • 감도맵 1회 추정 후 재사용(mask증강=그대로/flip=flip)은 network(varnet.py) 쪽 최적화 — 여기선 미포함.
      • 증강 확률 스케줄(delay→exp/ramp/constant)은 구현·epoch 전달(train_part set_epoch) 모두 배선 완료.
    """

    # GRAPPA 키(image 파일 내). load_data.py가 이 키로 읽어 넘겨준다.
    GRAPPA_KEY = "image_grappa"

    def __init__(self, isforward, max_key, augmentor=None, is_train=False, return_bbox=False,
                 normalize=False, readout_crop=False, crop_h=384, norm_percentile=99.0,
                 return_grappa=False, coil_compress=False, n_virtual_coils=8,
                 return_fg_mask=False):
        self.isforward = isforward
        self.max_key = max_key
        self.augmentor = augmentor
        self.is_train = bool(is_train) and (not isforward)
        self.return_bbox = bool(return_bbox)
        # ---- B5: 채점 정합 foreground 마스크 반환 (손실·검증을 몸통 영역으로 제한). 기본 off ----
        self.return_fg_mask = bool(return_fg_mask)
        # ---- 확정 전처리 플래그 (기본 off = 기존 동작 보존) ----
        self.normalize = bool(normalize)          # 전역 스칼라 ÷s (+ scale 반환)
        self.readout_crop = bool(readout_crop)    # readout 640→crop_h crop
        self.crop_h = int(crop_h)
        self.norm_percentile = float(norm_percentile)
        self.return_grappa = bool(return_grappa)  # GRAPPA 영상 prior 반환
        # ---- 효율 레버(후보): SVD coil compression. 기본 off. 배치 SME와 택1 ----
        #   켜면 코일 C(예 15)→n_virtual_coils(예 8)로 축소 → 반환 kspace shape의 coil축이 바뀜.
        #   화질은 neutral(속도/VRAM 레버). 상세: coil_compress() docstring, candidates.md.
        self.coil_compress = bool(coil_compress)
        self.n_virtual_coils = int(n_virtual_coils)

    def set_epoch(self, epoch):
        """train_part.py가 매 epoch 호출(스케줄 갱신). augmentor 없으면 no-op."""
        if self.augmentor is not None:
            self.augmentor.set_epoch(epoch)

    def _parse_bbox(self, attrs, sl):
        if not isinstance(attrs, dict):
            return []
        raw = attrs.get("annotations", None)
        if raw is None:
            return []
        try:
            d = json.loads(raw) if isinstance(raw, (str, bytes, bytearray)) else {}
        except (ValueError, TypeError):
            return []
        return d.get(str(int(sl)), [])

    def __call__(self, mask, input, target, attrs, fname, slice, grappa=None):
        if not self.isforward:
            maximum = attrs[self.max_key]
            tgt = np.asarray(target, dtype=np.float32)
        else:
            maximum = -1
            tgt = None

        mask1d = np.asarray(mask)
        kspace_c = np.asarray(input)           # complex (coils, H, W)
        orig_cdtype = kspace_c.dtype           # 증강(FFT/exp)이 complex128로 승격시키는 것 방지

        bbox = self._parse_bbox(attrs, slice) if (self.return_bbox and not self.isforward) else None

        # GRAPPA 영상 prior (fft 안 함). forward(추론) 경로는 load_data가 아직 미제공 → None 유지.
        grappa_arr = None
        if self.return_grappa and (grappa is not None):
            grappa_arr = np.asarray(grappa, dtype=np.float32)

        if self.is_train and (self.augmentor is not None) and (tgt is not None):
            kspace_c, mask1d, tgt, bbox, grappa_arr = self.augmentor(
                kspace_c, mask1d, tgt, bbox, grappa_arr, fname, slice)
        else:
            kspace_c = kspace_c * (mask1d >= 1).astype(mask1d.dtype)  # 기존 동작 유지(val/forward)

        # (B5) foreground 마스크 — 채점 정의(metrics.foreground_mask)를 그대로 적용.
        #   ⚠️ ÷s 정규화 전의 raw-scale target에서 계산해야 임계값(2e-5, 절대 스케일)이 채점과 일치.
        #   증강(flip) 후·정규화 전 시점의 tgt(384²)에서 뽑는다. 이진·기하라 이후 정규화 공간에도 유효.
        fg_mask_arr = None
        if self.return_fg_mask and (tgt is not None):
            fg_mask_arr = foreground_mask(tgt).astype(np.float32)

        # (1) readout crop 640→crop_h (높이만, 폭 native). flip/mask와 commute라 순서 무관.
        if self.readout_crop:
            kspace_c = readout_crop(kspace_c, self.crop_h)

        # (1b) SVD coil compression C→n_virtual_coils. crop 뒤(작아진 뒤)에 SVD → 더 쌈.
        #      코일축 선형결합이라 crop과 commute. mask는 ACS 판정에만 쓰고 불변.
        #      미샘플 라인(0) 보존 → DC 그대로 성립. GRAPPA(코일결합 384²)는 무관.
        if self.coil_compress:
            kspace_c = coil_compress(kspace_c, mask1d, self.n_virtual_coils)

        # (2) 전역 스칼라 정규화 ÷s (입력유도, k-space·GRAPPA·target·maximum 동일 s). 출력단 ×s.
        #     압축 뒤에 계산 → global_scale의 RSS가 가상코일 기준 → 네트워크 입력과 일관.
        scale = None
        if self.normalize:
            scale = global_scale(kspace_c, self.norm_percentile)
            kspace_c = kspace_c / scale
            if tgt is not None:
                tgt = tgt / scale
            if grappa_arr is not None:
                grappa_arr = grappa_arr / scale
            if maximum != -1:
                maximum = float(maximum) / scale     # SSIM data_range 정합(정규화 공간)

        kspace_c = kspace_c.astype(orig_cdtype, copy=False)  # dtype 보존(모델은 float32 기대)
        kspace = to_tensor(kspace_c)
        kspace = torch.stack((kspace.real, kspace.imag), dim=-1)
        mask_t = torch.from_numpy(
            (mask1d >= 1).reshape(1, 1, kspace.shape[-2], 1).astype(np.float32)
        ).byte()

        target_out = -1 if self.isforward else to_tensor(tgt)

        # 핵심 6개(항상 고정) + 옵션(순서 고정: scale → grappa → bbox). 상세는 클래스 docstring.
        out = [mask_t, kspace, target_out, maximum, fname, slice]
        if self.normalize:
            out.append(np.float32(scale))
        if self.return_grappa:
            out.append(to_tensor(grappa_arr) if grappa_arr is not None else torch.empty(0))
        if self.return_fg_mask and not self.isforward:   # bbox보다 앞(고정크기 텐서 → 기본 collate). forward엔 target 없어 미포함.
            out.append(to_tensor(fg_mask_arr))
        if self.return_bbox:
            out.append(bbox if bbox is not None else [])
        return tuple(out)
