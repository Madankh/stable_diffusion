import torch
import torch.nn as nn
import math
from einops.layers.torch import Rearrange
from einops import rearrange
from utils import default

class ConvNextBlock(nn.Module):
    def __init__(
            self,
            in_channels, 
            out_channels, 
            mult=2, 
            time_embedding_dim=None, 
            norm=True, 
            group=8):
        super().__init__()
        self.mlp = (
            nn.Sequential(
                nn.GELU(),
                nn.Linear(time_embedding_dim, in_channels)
            )
            if time_embedding_dim
            else None
        )

        self.in_conv = nn.Conv2d(in_channels, in_channels, 7, padding=3,groups=in_channels)
        self.block = nn.Seqential(
            nn.GroupNorm(1,in_channels) if norm else nn.Identity(),
            nn.Conv2d(in_channels, out_channels * mult, 3, padding=1),
            nn.GELU(),
            nn.GroupNorm(1, out_channels*mult),
            nn.Conv2d(out_channels * mult , out_channels, 3, padding=1)
        )

        self.residual_conv = (
            nn.Conv2d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

        