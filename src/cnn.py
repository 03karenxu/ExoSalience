from typing import Optional, Sequence
import torch
import torch.nn as nn
from torch import Tensor


def conv_block(c_in: int, c_out: int, kernel_size: int) -> nn.Sequential:
    """
    two convolutions, then a max pool that halves the length

    c_in: channels coming in, c_out: channels going out (the number of filters in each conv)
    """
    pad = kernel_size // 2  # keeps the length unchanged through each conv
    return nn.Sequential(
        nn.Conv1d(c_in, c_out, kernel_size=kernel_size, padding=pad),
        nn.ReLU(),
        nn.Conv1d(c_out, c_out, kernel_size=kernel_size, padding=pad),
        nn.ReLU(),
        nn.MaxPool1d(kernel_size=2, stride=2),
    )


class CNN1DEncoder(nn.Module):
    """
    convolutional feature extractor for one view

    one conv_block per entry in `channels`: each block halves the curve's length while the channel count grows,
    then everything is flattened into one long vector so the head still knows where each pattern was
    """

    def __init__(self, channels: Sequence[int] = (16, 32, 64, 128), kernel_size: int = 5):
        super().__init__()
        blocks, c_in = [], 1  # the input is one curve, so one channel
        for c_out in channels:
            blocks.append(conv_block(c_in, c_out, kernel_size))
            c_in = c_out
        self.features = nn.Sequential(*blocks)
        self.flatten = nn.Flatten()  # [B, channels[-1], length] -> [B, channels[-1] * length]

    def forward(self, x):
        return self.flatten(self.features(x))


class AstronetMVP(nn.Module):
    """
    2 branch CNN for global + local (+ tabular, unused) -> outputs logits
    """
    def __init__(self, global_channels: Sequence[int] = (16, 32, 64, 128), local_channels: Sequence[int] = (16, 32),
                 kernel_size: int = 5, head_hidden: int = 256, dropout: float = 0.3, tabular_dim: int = 0):
        super().__init__()
        # the local view is 10x shorter (201 vs 2001 points), so it gets fewer blocks
        self.enc_g = CNN1DEncoder(global_channels, kernel_size)
        self.enc_l = CNN1DEncoder(local_channels, kernel_size)

        # hidden layer + ReLU lets the head combine evidence ("dip here AND no dip there");
        # LazyLinear works out its input size (both flattened branches) on the first forward pass
        self.head = nn.Sequential(
            nn.LazyLinear(head_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, 1),
        )

    def forward(self, x_g, x_l, x_tab: Optional[Tensor]):  # not using tab for now
        f = torch.cat([self.enc_g(x_g), self.enc_l(x_l)], dim=1)
        return self.head(f)
