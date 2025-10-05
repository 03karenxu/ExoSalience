from typing import Optional
import torch
import torch.nn as nn
from torch import Tensor

class CNN1DEncoder(nn.Module):
    """
    convolutional feature extractor from 1d vector
    """

    def __init__(self, hidden=64, kernel_size=5):
        super().__init__()
        pad = kernel_size // 2
        self.features = nn.Sequential(
            nn.Conv1d(1, hidden, kernel_size=kernel_size, padding=pad),
            nn.ReLU(),
            nn.Conv1d(hidden, hidden, kernel_size=kernel_size, padding=pad),
            nn.ReLU()
        )
        # global avg pool over time
        self.gap = nn.AdaptiveAvgPool1d(1)   # -> [B, hidden, 1]

    def forward(self, x, return_feature_map: bool = False):
        z = self.features(x)
        z_gap = self.gap(z)
        z_embed = z_gap.squeeze(-1)
        if return_feature_map:
            return z_embed, z
        return z_embed
    
class AstronetMVP(nn.Module):
    """
    2 branch CNN for global + local + tabular -> outputs logits
    """
    def __init__(self, hidden=64, k_global=7, k_local=5, tabular_dim:int = 0):
        super().__init__()
        self.enc_g = CNN1DEncoder(hidden=hidden, kernel_size=k_global)
        self.enc_l = CNN1DEncoder(hidden=hidden, kernel_size=k_local)


        # lin reg. model head
        self.head = nn.LazyLinear(1) #TODO: add ReLU

    def forward(self, x_g, x_l, x_tab:Optional[Tensor], *, return_feature_maps: bool = False): # not using tab for now
        # feed global & local
        need_maps = return_feature_maps
        enc_g_out = self.enc_g(x_g, return_feature_map=need_maps)
        enc_l_out = self.enc_l(x_l, return_feature_map=need_maps)

        if need_maps:
            fg, fmap_g = enc_g_out
            fl, fmap_l = enc_l_out
        else:
            fg = enc_g_out
            fl = enc_l_out

        # concat global + local
        f = torch.cat([fg,fl],dim=1)

        # feed to lin reg model
        logits = self.head(f)
        if return_feature_maps:
            return logits, fg, fl, fmap_g, fmap_l
        return logits
    
