import json
import numpy as np
import torch


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
#   스케줄: 지수형 확률(첫 warmup epoch off → max_prob 점증) · 샘플별 결정적 RNG(재현성).
#   ※ epoch 전달·args 정의는 train_part.py/train.py 배선 필요(strategy/todo.md).
#     기본값은 전부 off라, 배선 전엔 기존 학습과 동일하게 동작.
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


class AugmentConfig:
    """증강 설정. train.py args에서 채워지며(없으면 getattr 기본값), 기본은 전부 off."""

    def __init__(self, **kw):
        self.on = kw.get("on", False)
        # 스케줄
        self.schedule = kw.get("schedule", "exp")          # 'exp' | 'constant'
        self.warmup_epochs = kw.get("warmup_epochs", 10)
        self.ramp_epochs = kw.get("ramp_epochs", 15)
        self.max_prob = kw.get("max_prob", 0.5)
        # 개별 aug on/off
        self.flip = kw.get("flip", True)                   # 좌-우(width)만
        self.mask = kw.get("mask", True)
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

    # ---- 스케줄 ----
    def _prob(self):
        c = self.cfg
        if not c.on:
            return 0.0
        if c.schedule == "constant":
            return c.max_prob
        e = self.epoch
        if e < c.warmup_epochs:                # warmup 동안 증강 off (기본기 학습)
            return 0.0
        t = (e - c.warmup_epochs) / max(1, c.ramp_epochs)
        return float(c.max_prob * (1.0 - np.exp(-3.0 * t)))  # 지수형 점증(t=1에서 ≈95%)

    def _rng(self, fname, sl):
        """샘플별 결정적 RNG: (seed, epoch, fname, slice) → aug ON/OFF A/B가 동일 추출 위에서 비교됨."""
        h = hash((self.cfg.seed, self.epoch, str(fname), int(sl))) & 0xFFFFFFFF
        return np.random.default_rng(h)

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

    def _flip_lr(self, K, target, bbox, mask):
        """좌-우(width) 플립을 이미지 도메인에서 수행 후 k-space 재생성 → 재시뮬 없이 픽셀 정확 정렬.
        flip은 center-crop과 commute하므로 384 target(=flip(clean))과 완전 일치.
        마스크는 even-length 정확 공식 M'[j]=M[(W-j)%W] = roll(flip(M),1)로 도출(1-라인 오차 방지).
        검증: recon(K_flip) == flip(recon(K)) (오차 ~1e-7), 샘플 라인 수 보존."""
        coil = _ifftc(K)[:, :, ::-1]
        K2 = _fftc(np.ascontiguousarray(coil))
        M2 = np.roll(np.flip(mask), 1).astype(np.float32)
        K2 = K2 * M2                                       # FFT 수치 누설 제거
        tgt = np.ascontiguousarray(target[:, ::-1])
        if bbox is not None:
            W = tgt.shape[1]
            bbox = [{**b, "x": W - int(b["x"]) - int(b["width"])} for b in bbox]
        return K2, tgt, M2, bbox

    def __call__(self, kspace_c, mask, target, bbox, fname, sl):
        """kspace_c:(coils,H,W) complex(raw) · mask:(W,) · target:(384,384) or None.
        반환: (kspace_masked_complex, mask_1d, target, bbox)."""
        cfg = self.cfg
        M = (mask >= 1).astype(mask.dtype)
        K = kspace_c * M                       # 제공 언더샘플 적용
        if (not cfg.on) or (target is None):
            return K, M, target, bbox
        p = self._prob()
        if p <= 0.0:
            return K, M, target, bbox
        rng = self._rng(fname, sl)
        tgt = target

        # 1) mask augmentation (원본 유지 확률 반영 → test 고정마스크 분포 정합)
        if cfg.mask and (rng.random() < p) and (rng.random() >= cfg.orig_mask_prob):
            M = self._aug_mask(rng, M)
            K = kspace_c * M

        # 2) 좌-우(width) flip — target·bbox·mask 동기(정렬 유지)
        if cfg.flip and (rng.random() < p):
            K, tgt, M, bbox = self._flip_lr(K, tgt, bbox, M)

        # 3) k-space 랜덤 위상 (global, magnitude/RSS 불변 → bbox 채점 불변, 이득 marginal)
        if cfg.phase and (rng.random() < p):
            K = K * np.exp(1j * rng.uniform(0.0, 2.0 * np.pi))

        # 4) 측정 노이즈 (보수적, 샘플된 라인에만)
        if cfg.noise and (cfg.noise_sigma > 0.0) and (rng.random() < p):
            scale = cfg.noise_sigma * float(np.abs(K).mean())
            n = (rng.standard_normal(K.shape) + 1j * rng.standard_normal(K.shape)) * scale
            K = K + n * M

        return K, M.astype(mask.dtype), np.ascontiguousarray(tgt), bbox


class DataTransform:
    def __init__(self, isforward, max_key, augmentor=None, is_train=False, return_bbox=False):
        self.isforward = isforward
        self.max_key = max_key
        self.augmentor = augmentor
        self.is_train = bool(is_train) and (not isforward)
        self.return_bbox = bool(return_bbox)

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

    def __call__(self, mask, input, target, attrs, fname, slice):
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

        if self.is_train and (self.augmentor is not None) and (tgt is not None):
            kspace_c, mask1d, tgt, bbox = self.augmentor(kspace_c, mask1d, tgt, bbox, fname, slice)
        else:
            kspace_c = kspace_c * (mask1d >= 1).astype(mask1d.dtype)  # 기존 동작 유지

        kspace_c = kspace_c.astype(orig_cdtype, copy=False)  # dtype 보존(모델은 float32 기대)
        kspace = to_tensor(kspace_c)
        kspace = torch.stack((kspace.real, kspace.imag), dim=-1)
        mask_t = torch.from_numpy(
            (mask1d >= 1).reshape(1, 1, kspace.shape[-2], 1).astype(np.float32)
        ).byte()

        target_out = -1 if self.isforward else to_tensor(tgt)

        if self.return_bbox:
            return mask_t, kspace, target_out, maximum, fname, slice, (bbox if bbox is not None else [])
        return mask_t, kspace, target_out, maximum, fname, slice
