from typing import Optional, Sequence, Union
import torch
from torch.utils.data import Dataset
from torch import Tensor

class ExoplanetDataset(Dataset):
    """
    Holds:
     - global curves for the global cnn embed
     - local curves for the local cnn embed
     - optional tabular (meta)data
     - labels

     Must be properly zip-able
    """

    def __init__(self, global_curves:Tensor, local_curves:Tensor, labels:Tensor, ids:Sequence[Union[int,str]], tabular:Optional[Tensor]=None):
        
        # length checks
        N = len(labels)
        assert len(global_curves) == len(local_curves) == N, "length mismatch"
        assert len(ids) == N, "length mismatch (ids)"

        if tabular is not None:
            assert len(tabular) == N, "length mismatch (tabular)"


        # shape checks
        assert global_curves.ndim == 2 and global_curves.shape[1] == 2001, "global curves wrong shape"
        assert local_curves.ndim == 2 and local_curves.shape[1] == 201, "local curves wrong shape"

        if tabular is not None:
            assert tabular.ndim == 2, "tabular wrong shape"

        assert labels.ndim in (1,2), "labels wrong shape"

        # store the tensors
        self.xg  = global_curves
        self.xl  = local_curves
        self.xt  = tabular

        # ensure binary targets for BCE loss (any label >0 becomes 1)
        labels = labels.float()
        if labels.ndim > 1:
            labels = labels.squeeze()
        self.y   = (labels > 0).float()

        self.ids = ids

    def __len__(self) -> int: return self.y.shape[0]

    def __getitem__(self, i:int):

        # get items for global lc, local lc, and tabular
        xg = self.xg[i].float()
        xl = self.xl[i].float()
        xt = None if self.xt is None else self.xt[i].float()
        y = self.y[i].float() #scalar float

        return xg, xl, xt, y, self.ids[i]

def collate(batch):
    xg, xl, xt, y, ids = zip(*batch)

    xg = torch.stack(xg, 0).unsqueeze(1)  # (B, 1, 2001)
    xl = torch.stack(xl, 0).unsqueeze(1)  # (B, 1, 201)
    y  = torch.stack(y,  0)

    if xt[0] is None:
        xt = None
    else:
        xt = torch.stack(xt, 0)

    return xg, xl, xt, y, list(ids)
