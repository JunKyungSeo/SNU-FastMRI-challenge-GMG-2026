"""MoE 라우터 — acc4/acc8 전문가를 마스크로 라우팅 (추론 전용).

채점 하네스(recon_eval.py, 고정)는 슬라이스마다 model(kspace, mask)를 호출하고 마스크를 넘긴다.
따라서 파일명·폴더명 없이 마스크의 샘플링 비율만으로 가속률(4/8)을 판별해 전문가를 고른다.
슬라이스당 전문가 1개만 실행하므로 추론시간은 단일 모델과 사실상 동일하다.

학습은 각 전문가를 따로 하고(train.py --acc 4 / --acc 8), 이 래퍼는 두 전문가를 합쳐
추론할 때만 쓴다(test_part.load_model이 결합 체크포인트를 보고 생성).
"""
import torch
import torch.nn as nn

# acc4 ≈ frac 0.25, acc8 ≈ frac 0.125 → 두 분포의 중앙(0.1875)을 임계값으로.
DEFAULT_ACC_THRESHOLD = 0.1875


def extract_acc_from_mask(mask, threshold=DEFAULT_ACC_THRESHOLD):
    """mask(임의 형상, PE축에 0/1) → 8 또는 4. frac=평균 샘플링 비율.
    frac < threshold 이면 더 많이 언더샘플된 acc8, 아니면 acc4."""
    frac = mask.float().mean().item()
    return 8 if frac < threshold else 4


class MoEWrapper(nn.Module):
    """(masked_kspace, mask) -> image. 마스크로 acc를 판별해 전문가 하나에 위임.

    recon_slice(test_part)가 model.readout_crop / crop_h / coil_compress / n_virtual_coils 를
    getattr로 읽어 전처리하므로, load_model이 이 속성들을 래퍼에 심어준다(두 전문가 동일 config)."""

    def __init__(self, expert4, expert8, threshold=DEFAULT_ACC_THRESHOLD):
        super().__init__()
        self.expert4 = expert4
        self.expert8 = expert8
        self.threshold = threshold

    def forward(self, masked_kspace, mask):
        acc = extract_acc_from_mask(mask, self.threshold)
        expert = self.expert8 if acc == 8 else self.expert4
        return expert(masked_kspace, mask)
