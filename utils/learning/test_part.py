import h5py
import numpy as np
import torch

from collections import defaultdict
from utils.common.utils import save_reconstructions
from utils.data.load_data import create_data_loaders
from utils.data.transforms import to_tensor
from utils.model.varnet import VarNet

# ---------------------------------------------------------------------------
# recon_eval.py(고정 타이밍 하네스)가 호출하는 팀-편집 계약: 아래 4개 심볼만 부른다.
# load_model은 체크포인트에 저장된 args에서 아키텍처를 읽어 SDUM/VarNet을 재구성한다
# → recon_eval.py는 수정 불필요(고정), num_cascades/channels 수동 매칭도 불필요.
# prep_volume/recon_slice는 모델 무관(kspace+mask 공급)이라 LIST varnet_phase2와 동일.
# ---------------------------------------------------------------------------
INPUT_KIND = "kspace"


def load_model(args, device):
    ckpt = torch.load(args.exp_dir / 'best_model.pt', map_location='cpu', weights_only=False)
    saved = ckpt.get('args', args)  # 학습 때 저장된 args(model/num_cascades/channels 포함)
    if getattr(saved, 'model', 'varnet') == 'sdum':
        from utils.model.sdum.config import tiny_8gb_args
        from utils.model.sdum.adapter import build_sdum
        model = build_sdum(tiny_8gb_args(
            num_cascades=saved.num_cascades, channels=list(saved.channels),
            uc=getattr(saved, "uc", False)))
    else:
        model = VarNet(
            num_cascades=getattr(saved, 'cascade', args.cascade),
            chans=getattr(saved, 'chans', args.chans),
            sens_chans=getattr(saved, 'sens_chans', args.sens_chans))
    model.load_state_dict(ckpt['model'])
    model.eval()
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
    kspace = to_tensor(ctx["kspace"][s] * mask)
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

    if getattr(args, "model", "varnet") == "sdum":
        from utils.model.sdum.config import tiny_8gb_args
        from utils.model.sdum.adapter import build_sdum
        sdum_args = tiny_8gb_args(num_cascades=args.num_cascades, channels=list(args.channels),
                                  uc=getattr(args, "uc", False))
        model = build_sdum(sdum_args)
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