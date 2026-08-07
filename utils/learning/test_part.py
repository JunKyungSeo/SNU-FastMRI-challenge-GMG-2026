import copy
import h5py
import numpy as np
import torch

from collections import defaultdict
from utils.common.utils import save_reconstructions
from utils.data.load_data import create_data_loaders
from utils.data.transforms import to_tensor, coil_compress, readout_crop
from utils.model.varnet import VarNet

# ---------------------------------------------------------------------------
# recon_eval.py(고정 타이밍 하네스)가 호출하는 팀-편집 계약: 아래 4개 심볼만 부른다.
# load_model은 체크포인트에 저장된 args에서 아키텍처를 읽어 SDUM/VarNet을 재구성한다
# → recon_eval.py는 수정 불필요(고정), num_cascades/channels 수동 매칭도 불필요.
# prep_volume/recon_slice는 모델 무관(kspace+mask 공급)이라 LIST varnet_phase2와 동일.
# ---------------------------------------------------------------------------
INPUT_KIND = "kspace"


def _set_preproc(model, saved):
    """전처리 정합: 학습과 동일 설정(readout crop·coil compression)을 ckpt args에서 읽어 recon_slice가 재적용.
    ⚠️ 필수: 학습이 --readout-crop(H 640→384)·--coil-compress를 쓰면 추론도 같은 입력이어야 한다
       (PromptMR는 비선형 → 입력 높이·코일수가 다르면 학습분포와 어긋나 성능 저하)."""
    model.readout_crop = bool(getattr(saved, 'readout_crop', False))
    model.crop_h = int(getattr(saved, 'crop_h', 384))
    model.coil_compress = bool(getattr(saved, 'coil_compress', False))
    model.n_virtual_coils = int(getattr(saved, 'n_virtual_coils', 8))
    # 2.5D 인접 슬라이스도 학습과 동일해야 한다 — 학습이 3슬라이스면 추론도 3슬라이스를 쌓아야
    # 코일축 크기가 맞고(안 맞으면 conv in_chans 불일치로 즉시 에러), 중앙 슬라이스 규약도 성립한다.
    model.num_adj_slices = int(getattr(saved, 'num_adj_slices', 1))
    return model


def _build_expert(saved, state_dict, default_args):
    """저장된 args로 아키텍처를 재구성하고 가중치를 로드(CPU, eval). 단일 모델·MoE 전문가 공용."""
    if getattr(saved, 'model', 'varnet') == 'promptmr':
        from utils.model.promptmr.adapter import build_promptmr
        model = build_promptmr(saved)
        model.use_checkpoint = False   # 추론: gradient checkpointing 불필요
    else:
        model = VarNet(
            num_cascades=getattr(saved, 'cascade', default_args.cascade),
            chans=getattr(saved, 'chans', default_args.chans),
            sens_chans=getattr(saved, 'sens_chans', default_args.sens_chans))
    _set_preproc(model, saved)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def load_model(args, device):
    ckpt = torch.load(args.exp_dir / 'best_model.pt', map_location='cpu', weights_only=False)

    # MoE 결합 체크포인트: acc4/acc8 전문가를 마스크로 라우팅. recon_eval는 그대로(net_name 1개).
    if ckpt.get('moe', False):
        from utils.model.moe import MoEWrapper, DEFAULT_ACC_THRESHOLD
        experts = ckpt['experts']
        # ⚠️ deepcopy 필수. promptmr_plus.py:277의 act=nn.PReLU()는 mutable default argument라
        # import 시점에 만들어진 PReLU **하나**를 모든 CAB이, 심지어 **다른 모델 인스턴스끼리도**
        # 공유한다. 그대로 두면 나중에 로드한 acc8의 slope가 acc4 전문가까지 덮어써서
        # acc4가 acc8의 활성화로 돈다. deepcopy는 모델 **내부** 공유(감도맵 이중 등록 등)는
        # 보존하면서 인스턴스 **바깥** 공유만 끊는다. 근본 수정(act=None)은 학습 의미가 바뀌어
        # (전역 slope 1개 → CAB마다 1개) 기존 체크포인트와 비교가 깨지므로 대회 후로 미룬다.
        e4 = copy.deepcopy(_build_expert(experts['acc4']['args'], experts['acc4']['model'], args))
        e8 = _build_expert(experts['acc8']['args'], experts['acc8']['model'], args)
        threshold = ckpt.get('router', {}).get('threshold', DEFAULT_ACC_THRESHOLD)
        model = MoEWrapper(e4, e8, threshold=threshold)
        # recon_slice가 읽는 전처리 속성을 래퍼에 심는다(두 전문가 동일 config 전제 — 병합 스크립트가 검증).
        _set_preproc(model, experts['acc4']['args'])
        model.eval()
        return model.to(device=device)

    saved = ckpt.get('args', args)  # 학습 때 저장된 args(model/num_cascades/n_feat0 등 포함)
    model = _build_expert(saved, ckpt['model'], args)
    return model.to(device=device)


def prep_volume(image_path, kspace_path, device):
    """한 볼륨의 k-space와 mask를 호스트로 로드. 시간측정 제외(모델 연산 없음)."""
    with h5py.File(kspace_path, 'r') as hf:
        kspace = hf['kspace'][:]
        mask = np.array(hf['mask'])
    return {"kspace": kspace, "mask": mask, "device": device, "num_slices": kspace.shape[0]}


def recon_slice(model, ctx, s):
    """슬라이스 1장 복원(batch=1). 하네스가 이 호출의 시간만 잰다."""
    device = ctx["device"]
    mask = ctx["mask"]
    adj = int(getattr(model, "num_adj_slices", 1))
    if adj == 1:
        ks = ctx["kspace"][s] * mask
    else:
        # 학습(SliceData)과 **동일 규약**: 코일축에 슬라이스-major로 쌓고 볼륨 경계는 clamp.
        n, half = ctx["num_slices"], adj // 2
        idxs = [min(max(s + d, 0), n - 1) for d in range(-half, half + 1)]
        ks = np.concatenate([ctx["kspace"][j] for j in idxs], axis=0) * mask
    # 학습(DataTransform)과 동일 순서·설정으로 전처리 재적용 → train/infer 정합.
    #   ① readout crop(H만 640→crop_h, 무손실)  ② coil compression(ACS SVD)  — DataTransform과 동일 순서.
    #   readout_crop은 axis -2(H)만 건드려 코일축에 쌓인 슬라이스 전부에 동일 적용된다(슬라이스별과 동일).
    if getattr(model, "readout_crop", False):
        ks = readout_crop(ks, getattr(model, "crop_h", 384))
    if getattr(model, "coil_compress", False):
        ks = coil_compress(ks, np.ravel(mask), getattr(model, "n_virtual_coils", 8))
    ks = ks.astype(ctx["kspace"].dtype)   # readout_crop/coil_compress의 complex128 승격 방어
    kspace = to_tensor(ks)
    kspace = torch.stack((kspace.real, kspace.imag), dim=-1).unsqueeze(0).to(device=device)
    mask_t = torch.from_numpy(mask.reshape(1, 1, kspace.shape[-2], 1).astype(np.float32)).byte()
    mask_t = mask_t.unsqueeze(0).to(device=device)
    return model(kspace, mask_t)[0]


def test(args, model, data_loader):
    model.eval()
    reconstructions = defaultdict(dict)
    
    with torch.no_grad():
        for (mask, kspace, _, _, fnames, slices) in data_loader:
            kspace = kspace.cuda(non_blocking=True)
            mask = mask.cuda(non_blocking=True)
            output = model(kspace, mask)

            for i in range(output.shape[0]):
                reconstructions[fnames[i]][int(slices[i])] = output[i].cpu().numpy()

    for fname in reconstructions:
        reconstructions[fname] = np.stack(
            [out for _, out in sorted(reconstructions[fname].items())]
        )
    return reconstructions, None


def forward(args):

    device = torch.device(f'cuda:{args.GPU_NUM}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)
    print ('Current cuda device ', torch.cuda.current_device())

    if getattr(args, "model", "varnet") == "promptmr":
        from utils.model.promptmr.adapter import build_promptmr
        model = build_promptmr(args)
        model.use_checkpoint = False   # 추론: gradient checkpointing 불필요
    else:
        model = VarNet(num_cascades=args.cascade,
                       chans=args.chans,
                       sens_chans=args.sens_chans)
    model.to(device=device)
    
    checkpoint = torch.load(args.exp_dir / 'best_model.pt', map_location='cpu', weights_only=False)
    print(checkpoint['epoch'], checkpoint['best_val_loss'].item())
    model.load_state_dict(checkpoint['model'])
    
    forward_loader = create_data_loaders(data_path = args.data_path, args = args, isforward = True)
    reconstructions, inputs = test(args, model, forward_loader)
    save_reconstructions(reconstructions, args.forward_dir, inputs=inputs)