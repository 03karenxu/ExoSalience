import torch
from typing import Optional, Union, Sequence
import numpy as np

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "model_scripted.pt"
model = torch.jit.load(MODEL_PATH, map_location=DEVICE)
model.eval()

def _to_tensor(x: Union[torch.Tensor, np.ndarray, Sequence], dtype=torch.float32, device=DEVICE):
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype)
    return torch.as_tensor(x, dtype=dtype, device=device)

def _prep_curve(x):
    t = _to_tensor(x)
    if t.dim() == 1:
        t = t.unsqueeze(0).unsqueeze(0)
    elif t.dim() == 2:
        t = t.unsqueeze(1)
    elif t.dim() == 3:
        pass
    else:
        raise ValueError(f"curve must be 1/2/3-D, got {t.shape}")
    return t.contiguous()

def _prep_tab(x):
    t = _to_tensor(x)
    if t.dim() == 1:
        t = t.unsqueeze(0)
    elif t.dim() == 2:
        pass
    else:
        raise ValueError(f"tabular must be 1/2-D, got {t.shape}")
    return t.contiguous()

@torch.inference_mode()
def infer(
    x_global,
    x_local,
    x_tab: Optional[Union[torch.Tensor, np.ndarray, Sequence]] = None,
    return_feature_maps: bool = False,
):
    """
    x_global: [T] | [B,T] | [B,1,T]
    x_local : [T] | [B,T] | [B,1,T]
    x_tab   : optional [F] | [B,F]
    returns: list of probabilities length B
    """
    xg = _prep_curve(x_global)
    xl = _prep_curve(x_local)
    if x_tab is None:
        logits, fg, fl, fmap_g, fmap_l = model(xg, xl)
    else:
        xt = _prep_tab(x_tab)
        logits, fg, fl, fmap_g, fmap_l = model(xg, xl, xt)

    probs = torch.sigmoid(logits).squeeze(-1)

    if not return_feature_maps:
        return probs.detach().cpu().tolist()

    return {
        "logits": logits.detach().cpu(),
        "probs": probs.detach().cpu(),
        "global_embed": fg.detach().cpu(),
        "local_embed": fl.detach().cpu(),
        "global_feature_map": fmap_g.detach().cpu(),
        "local_feature_map": fmap_l.detach().cpu(),
    }
