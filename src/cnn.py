import torch
import torch.nn as nn

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

    def forward(self, x):
        z = self.features
        z = self.gap(z)
        z = z.squeeze(-1)
        return z
    
class AstronetMVP(nn.Module):
    """
    2 branch CNN for global + local + tabular -> outputs logits
    """
    def __init__(self, hidden=64, k_global=7, k_local=5, tabular_dim:int = 0):
        super().__init__()
        self.enc_g = CNN1DEncoder(hidden=hidden, kernel_size=k_global)
        self.enc_l = CNN1DEncoder(hidden=hidden, kernel_size=k_local)

        in_dim = hidden + hidden + (tabular_dim if tabular_dim > 0 else 0)

        # lin reg. model head
        self.head = nn.Linear(in_dim, 1) #TODO: add ReLU

    def forward(self, x_g, x_l, x_tab):
        # feed global & local
        fg = self.enc_g(x_g)
        fl = self.enc_l(x_l)

        # concat global + local
        f = torch.cat([fg,fl],dim=1)

        # feed to lin reg model
        logits = self.head(f)
        return logits
    
