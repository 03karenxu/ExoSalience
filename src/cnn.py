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
    