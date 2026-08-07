import json
import zlib
import cv2
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
# Data Augmentation (strategy/augmentation.md 구현) — 좌-우(width/phase-encode) 플립 하나뿐.
#   설계 원칙: hallucination-free lesion fidelity — bbox 안전 + 물리적으로 유효한 증강만.
#   2026-07-28 결정으로 mask thinning·k-space 위상·측정 노이즈를 **코드에서 제거**했다.
#     대회 입력이 acc4/acc8 두 가지로 고정이라 마스크를 더 솎으면 채점에 없는 가속률
#     (주변부 간격 12~32)을 학습하게 되고, MoE 라우터 기준(마스크 샘플 비율 0.25)으로는
#     acc4 샘플이 acc8 쪽 입력으로 바뀌어 전문가 분리 실험까지 흐린다.
#     위상은 magnitude 불변이라 marginal, 노이즈는 검증 전이었다. 근거: augmentation.md §0.
#   ⚠️ 이 파일 아래쪽의 **마스크 재생성(remask)은 위에서 지운 mask thinning이 아니다.** 생성
#     치역이 대회 마스크 가족과 동일하고(step 4|8 · cf 0.08, 학습풀+leaderboard 280볼륨 비트정확
#     재현), 학습 중인 전문가의 가속률과 마스크가 CLI로 강제 일치해 라우터도 안 흔든다.
#     지우기 전에 augmentation.md §0의 "마스크 재생성" 절을 읽을 것 — 한 번 지워진 전례가 있다.
#   미채택(기록): 위-아래(readout) 플립 = 무릎 뼈 위치가 뒤바뀌는 비현실 형태 (shell29 실측 상한 +0.003).
#   affine(회전±5°+확대≤1.1, 2026-08-07 채택): translation 미채택 사유였던 "384 crop과 정렬 불가"를
#     target 재생성(full k-space 워프 → RSS)으로 해소했다 — 아래 MRIAugmentor._affine 참고.
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
    ACS는 sensitivity map/GRAPPA 캘리브레이션 소스다 — coil_compress가 이 구간에서만
    압축행렬을 뽑는다(추론에서 완전샘플인 구간이 여기뿐이라서)."""
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
    "입력 유도(label 통계 금지)" 원칙과 동일한 계열). 남은 증강이 좌우 flip 하나뿐이고
    flip은 ACS를 그대로 옮기기만 하므로 증강과도 안전하게 호환된다.

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
    coil compression은 **배치 SME(model_strategy §3.2)와 택1(동시적용 금지)** — 둘 다
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


def _rss_crop(imgs, hw=(384, 384)):
    """코일 복소 영상 → RSS → 중앙 crop(부족한 축은 0-pad) → float32.
    affine 증강의 target 재생성 규약 — 제공 label과의 정합은
    scripts/verify_affine_target.py가 실데이터 전수로 확인한다(corr≥0.9999)."""
    rss = np.sqrt((np.abs(imgs) ** 2).sum(axis=0)).astype(np.float32)
    H, W = rss.shape
    th, tw = hw
    out = np.zeros(hw, dtype=np.float32)
    sh, dh = max((H - th) // 2, 0), max((th - H) // 2, 0)
    sw, dw = max((W - tw) // 2, 0), max((tw - W) // 2, 0)
    ch, cw = min(H, th), min(W, tw)
    out[dh:dh + ch, dw:dw + cw] = rss[sh:sh + ch, sw:sw + cw]
    return out


class AugmentConfig:
    """증강 설정. train.py args에서 채워지며(없으면 getattr 기본값), 기본은 off.
    증강이 좌우 flip 하나뿐이라 변환 선택 스위치·가중치 필드는 없다 — on/off는 `on` 하나."""

    def __init__(self, **kw):
        self.on = kw.get("on", False)
        # 스케줄 — MRAugment 공식 repo(mraugment/data_augment.py: schedule_p)와 동일 형태.
        #   p(t<D)=0, p(T)=p_max. T로 정규화하므로 런 길이가 바뀌어도 증강량이 스케일된다.
        self.schedule = kw.get("schedule", "exp")          # 'exp' | 'ramp' | 'constant'
        self.delay = kw.get("delay", 0)                    # D: 증강 없이 보내는 앞쪽 epoch 수
        self.total_epochs = kw.get("total_epochs", 1)      # T: args.num_epochs (load_data가 채움)
        self.exp_decay = kw.get("exp_decay", 5.0)          # 논문/공식 repo 기본 c=5.0
        self.max_prob = kw.get("max_prob", 0.55)           # 논문 기본 p_max=0.55
        # affine(회전+확대) — 2026-08-07 채택(사용자 확정: shear 없음·확대만·±5°).
        self.affine = kw.get("affine", True)               # --aug-on 아래 기본 on. False면 flip 추첨 스트림까지 구버전 동일
        self.rot_max = kw.get("rot_max", 5.0)              # 회전 최대각(도) — U(-rot_max, +rot_max)
        self.scale_max = kw.get("scale_max", 1.1)          # 확대 상한 — U(1.0, scale_max). 축소는 빈 띠가 생겨 안 쓴다
        # 변환별 최대확률(2026-08-08) — None이면 max_prob로 fallback(기존 셸 인자 100% 호환).
        # 0.0은 "그 변환만 확률 0"인 유효값이므로 `or` 대신 None 검사로 해석한다.
        v = kw.get("flip_max_prob", None)
        self.flip_max_prob = float(self.max_prob if v is None else v)
        v = kw.get("affine_max_prob", None)
        self.affine_max_prob = float(self.max_prob if v is None else v)
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
    def prob(self, p_max=None):
        """이번 epoch에서 증강이 걸릴 확률. 학습 로그와 __call__이 같은 값을 쓰도록 한 곳에서 계산.
        MRAugment 공식 schedule_p와 동일: D=delay, T=total_epochs, c=exp_decay/(T-D).
        p(t<D)=0이고 p(T)=p_max로 정규화되므로 런 길이가 달라도 증강량이 일관된다.
        (epoch은 0..T-1이라 마지막 epoch의 p는 p_max에 살짝 못 미친다 — 상류와 동일.)
        p_max 인자(2026-08-08): 변환별 도달점(cfg.flip_max_prob/affine_max_prob)으로 같은
        곡선을 스케일할 때 쓴다. None이면 cfg.max_prob(기존 동작 — 스케줄 테스트가 이 계약을 쓴다)."""
        c = self.cfg
        if not c.on:
            return 0.0
        D, T, t = int(c.delay), int(c.total_epochs), self.epoch
        p_max = float(c.max_prob if p_max is None else p_max)
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
        # closed form은 순수 인덱스 치환이라 K2의 support가 이미 정확히 M2다(수치 누설 없음).
        # 이 곱은 방어용 — 호출부가 마스크 안 씌운 K를 넘기더라도 계약(K2 ⊆ M2)을 지킨다.
        K2 = K2 * M2
        tgt = np.ascontiguousarray(target[:, ::-1])
        if bbox is not None:
            W = tgt.shape[1]
            bbox = [{**b, "x": W - int(b["x"]) - int(b["width"])} for b in bbox]
        g = None if grappa is None else np.ascontiguousarray(grappa[..., ::-1])
        return K2, tgt, M2, bbox, g

    def _affine(self, kspace_c, target_hw, rng):
        """회전 U(-rot_max,+rot_max)° + 확대 U(1,scale_max)를 '환자 통째로' 적용하고
        target을 재생성한다 (MRAugment 방식). 반환: (warped_kspace_c, new_target).

        왜 label 재사용이 아니라 재생성인가: label은 384² 고정 크롭이라 기하 변환 시
        프레임 밖 정보가 없다. 학습 h5의 kspace는 full(remask에서 170/170 실측)이므로
        코일별 IFFT→워프→FFT로 '변형된 환자의 획득'을 만들고 정답도 같은 워프 영상의
        RSS에서 뽑으면 입력·정답이 구성상 정합한다(label↔full recon RSS corr 1.0 — remask 검증).

        보간: INTER_LANCZOS4 — bilinear 대비 고주파 보존(테스트로 강제. 채점이 SSIM이라
        블러가 곧 손해다). 회전 모서리 공백은 0 채움 — 입력·정답이 같은 공백을 공유하므로
        오지도(誤指導)가 없다. 코일 감도는 환자와 '함께 도는' 셈이 되는데, 이는 다른 코일
        배치의 유효한 획득이고 감도는 sens_net이 슬라이스별로 재추정하므로 문제없다.
        ⚠️ 2.5D(인접 슬라이스 입력)를 켜는 날에는 '한 입력 안의 슬라이스들은 같은 변형'
        제약이 새로 생긴다 — 지금은 슬라이스 1장 입력이라 해당 없음."""
        deg = float(rng.uniform(-self.cfg.rot_max, self.cfg.rot_max))
        s = float(rng.uniform(1.0, self.cfg.scale_max))
        C, H, W = kspace_c.shape
        mat = cv2.getRotationMatrix2D(((W - 1) / 2.0, (H - 1) / 2.0), deg, s)
        imgs = _ifftc(kspace_c)
        out = np.empty_like(imgs)
        for c in range(C):
            ch2 = np.stack([imgs[c].real, imgs[c].imag], axis=-1).astype(np.float32, copy=False)
            w = cv2.warpAffine(ch2, mat, (W, H), flags=cv2.INTER_LANCZOS4,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
            out[c] = w[..., 0] + 1j * w[..., 1]
        return _fftc(out), _rss_crop(out, target_hw)

    def __call__(self, kspace_c, mask, target, bbox, grappa, fname, sl):
        """kspace_c:(coils,H,W) complex(raw full) · mask:(W,) · target:(384,384) or None
        · grappa:(384,384) magnitude or None.
        반환: (kspace_masked_complex, mask_1d, target, bbox, grappa).
        순서: affine(mask 곱 전, full k-space + target 재생성) → mask 곱 → flip(기존).
        affine이 안 걸리면 target은 제공 label 그대로다(재생성은 걸린 샘플만)."""
        M = (mask >= 1).astype(mask.dtype)
        if (not self.cfg.on) or (target is None):
            return kspace_c * M, M, target, bbox, grappa
        p_flip = self.prob(self.cfg.flip_max_prob)
        p_affine = self.prob(self.cfg.affine_max_prob) if self.cfg.affine else 0.0
        if p_flip <= 0.0 and p_affine <= 0.0:
            return kspace_c * M, M, target, bbox, grappa
        rng = self._rng(fname, sl)
        tgt = target

        # 1) affine(회전+확대) — full k-space에서. cfg.affine=False면 rng를 소비하지
        #    않아(and 단락) flip 추첨 스트림이 구버전과 비트 동일하게 유지된다.
        if self.cfg.affine and (rng.random() < p_affine):
            assert not bbox, "affine 증강은 bbox 변환 미구현 — --bbox-loss-lambda와 같이 켤 수 없다"
            assert grappa is None, "affine 증강은 GRAPPA 워프 미구현 — GRAPPA 입력과 같이 켤 수 없다"
            kspace_c, tgt = self._affine(kspace_c, (384, 384), rng)

        K = kspace_c * M
        # 2) 좌-우(width) flip — target·bbox·mask·GRAPPA 동기 (기존 그대로, 독립 추첨)
        if rng.random() < p_flip:
            K, tgt, M, bbox, grappa = self._flip_lr(K, tgt, bbox, M, grappa)
        return K, M.astype(mask.dtype), np.ascontiguousarray(tgt), bbox, grappa


# ============================================================================
# 마스크 재생성 (remask) — 학습 로더 전용. 근거: augmentation.md §0 후속(2026-07-29).
#   학습풀 h5의 kspace는 **마스크 곱 이전 full k-space**다(train 170/170 · val 30/30 실측).
#   그래서 딸려온 마스크를 버리고 우리가 만든 마스크를 곱하면 같은 볼륨을 반대 가속률로 쓸 수 있다.
#   정답 image_label은 fully-sampled 복원이라(corr 1.0000000000, attrs['max']도 label 기준)
#   마스크를 바꿔도 그대로 유효하다 → MoE 전문가 하나가 보는 볼륨이 80 → 160이 된다
#   (acc4 전문가는 acc8 파일에서 뽑은 acc4로, acc8 전문가는 acc4 파일에서 뽑은 acc8로).
#
#   생성기는 주최측 것과 동일하다 — 학습풀 200 + leaderboard 80 전수에서 center_fraction 후보
#   {0.04, 0.08, 0.10} 완전탐색 결과 **네 셀 모두 0.08만 유일하게** 비트정확 재현(실패 0건).
#   ⚠️ 벤더 fastmri의 EquispacedMaskFunc는 adjusted_accel(실수 간격)을 써서 이 데이터와 안 맞는다.
#      np.arange(offset, W-1, acc)도 금지 — (W-1)%acc==offset인 볼륨에서 마지막 열이 조용히 빠진다.
#      슬라이싱 m[offset::acc]만 이 데이터의 규약과 일치한다.
#
#   [offset] 학습풀은 offset == (W//2)%acc로 **결정론적**이다(200볼륨 전수, 예측=실측).
#   그런데 leaderboard는 **무작위 0~acc-1**이다(acc4 예측 24/0/16/0 vs 실측 15/11/7/7).
#   그래서 기본은 uniform이다 — 지금 acc8 전문가는 채점 볼륨의 65%를 학습에서 본 적 없는
#   offset으로 만난다. canonical은 "그 볼륨이 반대 가속률로 배정됐다면 주최측이 붙였을 바로
#   그 마스크"를 재현하는 모드로, 분포를 전혀 안 건드리고 볼륨 풀만 2배로 늘리고 싶을 때 쓴다.
# ============================================================================

CENTER_FRACTION = 0.08   # acc4·acc8 공통. 280볼륨 비트정확 재현으로 확정. CLI로 노출하지 말 것.


def equispaced_mask(W, acc, offset, center_fraction=CENTER_FRACTION, dtype=np.float32):
    """대회 마스크 생성기. mask = {i : i ≡ offset (mod acc)} ∪ ACS[pad, pad+nlf).
    nlf = round(cf·W)(W=368→29, W=372→30), pad = (W−nlf+1)//2(→170, →171)."""
    W, acc, offset = int(W), int(acc), int(offset)
    m = np.zeros(W, dtype=dtype)
    m[offset % acc::acc] = 1                    # ★ 슬라이싱 규약 = range(offset, W, acc)
    nlf = int(round(center_fraction * W))
    pad = (W - nlf + 1) // 2
    m[pad:pad + nlf] = 1
    return m


def canonical_offset(W, acc):
    """학습풀의 offset 규칙 — 격자가 k-space 중앙열 W//2를 지난다. 200볼륨 전수 일치."""
    return (int(W) // 2) % int(acc)


def remask_offset(seed, epoch, fname, sl, acc):
    """(seed, epoch, fname, slice)에서 결정적으로 뽑는 균등 offset.
    MRIAugmentor._rng와 같은 관례(zlib.crc32 — builtin hash는 프로세스마다 달라 재현 불가)이되
    salt "remask"로 스트림을 분리한다 → flip 동전 시퀀스가 remask on/off에서 동일해야 A/B가 성립."""
    key = repr(("remask", int(seed), int(epoch), str(fname), int(sl), int(acc))).encode()
    return int(np.random.default_rng(zlib.crc32(key)).integers(int(acc)))


# ---- 마스크 증강(--mask-aug) — 재생성 경로를 그대로 쓰되 무엇을 흔들지만 바꾼다 -------------
#   offset : 가속률은 그대로 두고 격자 위상만 매 epoch 재추첨(채점의 무작위 offset 분포에 맞춤).
#   full   : 가속률까지 4|8 중에서 재추첨 → 한 볼륨이 두 가속률을 오간다.
#            전문가는 한 가속률만 봐야 하므로 --remask 와는 배타(train.py에서 fail-fast).

def mask_aug_acc(seed, epoch, fname, sl):
    """--mask-aug full 에서 뽑는 가속률(4|8). remask_offset과 같은 관례이되 salt를 달리해
    스트림을 분리한다 — 안 나누면 가속률 추첨이 offset 추첨과 맞물려 조합이 편향된다."""
    key = repr(("mask_aug_acc", int(seed), int(epoch), str(fname), int(sl))).encode()
    return 4 if int(np.random.default_rng(zlib.crc32(key)).integers(2)) == 0 else 8


def infer_acc(mask1d):
    """딸려온 마스크에서 그 파일의 가속률을 읽는다 — ACS 밖 격자점의 **최소 간격**이 곧 acc
    (ACS를 가로지르는 한 쌍만 더 벌어지므로 min이면 안전).
    --mask-aug offset 이 --remask 없이 돌 때 "파일의 원래 가속률"을 유지하려고 쓴다.
    파일명 accN 토큰을 읽는 길도 있지만 그 규약은 load_data 쪽이고, 여기서 import 하면
    transforms↔load_data 순환이 된다. 마스크 자체가 더 직접적인 근거이기도 하다."""
    m = np.asarray(mask1d) >= 1
    idx = np.where(m & ~_detect_acs(m))[0]
    if idx.size < 2:
        raise ValueError("마스크에서 가속률을 못 읽었다 — ACS 밖 샘플 라인이 2개 미만")
    return int(np.diff(idx).min())


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
                 return_fg_mask=False, remask_acc=None, remask_offset="uniform",
                 remask_seed=430, mask_aug="off"):
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
        #   화질은 neutral(속도/VRAM 레버). 상세: coil_compress() docstring, strategy/preprocessing.md A2.
        self.coil_compress = bool(coil_compress)
        self.n_virtual_coils = int(n_virtual_coils)
        # ---- 마스크 재생성(학습 전용). None = off = 딸려온 마스크 그대로(기존과 비트 동일) ----
        self.remask_acc = None if remask_acc in (None, "off", "") else int(remask_acc)
        self.remask_offset = str(remask_offset)
        self.remask_seed = int(remask_seed)
        # ---- 마스크 증강(off|offset|full). off = 위 remask 로직 그대로(기존과 비트 동일) ----
        self.mask_aug = str(mask_aug)
        self.epoch = 0

    def set_epoch(self, epoch):
        """train_part.py가 매 epoch 호출(스케줄 갱신). augmentor 없으면 augmentor만 no-op.
        remask의 uniform offset도 이 epoch을 쓴다 — 안 받으면 매 epoch 같은 마스크가 되어
        크래시 없이 이득만 조용히 사라진다."""
        self.epoch = int(epoch)
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

        # 마스크 재생성 — 딸려온 마스크를 버리고 대회 생성기로 다시 만든다(학습 전용).
        #   full k-space라 어떤 가속률 마스크를 곱해도 주최측 입력 생성식과 같은 물건이 되고,
        #   정답은 마스크와 무관하다. 아래 언더샘플 곱(augmentor의 K=kspace*M 또는 else 분기)이
        #   새 마스크로 수행된다 — 이 파일에서 동작이 바뀌는 지점은 여기 하나뿐이다.
        #   val/forward는 is_train 게이트에 걸린다(load_data도 애초에 None을 넘긴다 = 2중 방어).
        #   --mask-aug 도 같은 경로를 탄다 — off가 아니면 --remask가 꺼져 있어도 재생성한다.
        #     offset: 가속률은 그대로(--remask 값, 없으면 딸려온 마스크에서 읽은 파일 원래 값) 두고 위상만 재추첨.
        #     full  : 가속률까지 4|8 재추첨(--remask와는 배타 — train.py에서 막는다).
        if ((self.remask_acc is not None) or self.mask_aug != "off") and self.is_train and (tgt is not None):
            W = kspace_c.shape[-1]
            if self.mask_aug == "full":
                acc = mask_aug_acc(self.remask_seed, self.epoch, fname, slice)
            elif self.remask_acc is not None:
                acc = self.remask_acc
            else:
                acc = infer_acc(mask1d)
            # mask_aug가 켜지면 위상 재추첨이 목적이라 canonical 고정을 덮는다.
            off = (canonical_offset(W, acc) if (self.mask_aug == "off" and self.remask_offset == "canonical")
                   else remask_offset(self.remask_seed, self.epoch, fname, slice, acc))
            mask1d = equispaced_mask(W, acc, off, dtype=mask1d.dtype)

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
