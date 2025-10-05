from typing import Optional
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

    def __init__(self, global_curves:Tensor, local_curves:Tensor, labels:Tensor, tabular:Optional[Tensor]=None, ids:Optional[tuple]=None):
        
        # length checks
        N = len(labels)
        assert len(global_curves) == len(local_curves) == N, "length mismatch"

        if tabular is not None:
            assert len(tabular) == N, "length mismatch (tabular)"

        if ids is not None:
            assert len(ids) == N, "length mismatch (ids)"

        # shape checks
        assert global_curves.ndim == 3 and global_curves.shape[1] == 1, "global curves wrong shape"
        assert local_curves.ndim == 3 and local_curves.shape[1] == 1, "local curves wrong shape"

        if tabular is not None:
            assert tabular.ndim == 2, "tabular wrong shape"

        assert labels.ndim in (1,2), "labels wrong shape"

        # store the tensors
        self.xg  = global_curves
        self.xl  = local_curves
        self.xt  = tabular

        # squeeze labels, cast to float for logits
        self.y   = labels.float()
        self.ids = ids # returned if provided (optional)

    def __len__(self) -> int: return self.y.shape[0]

    def __getitem__(self, i:int):

        # get items for global lc, local lc, and tabular
        xg = self.xg[i]
        xl = self.xl[i]
        xt = None if self.xt is None else self.xt[i]
        y = self.y[i] #scalar float

        if self.ids is None:
            return xg, xl, xt, y
        else: return xg, xl, xt, y, self.ids[i]
