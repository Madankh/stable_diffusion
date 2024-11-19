import math
from functools import partial
import numpy as np
import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange

# Helpers functions
def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable() else d

def cast_tuple(t, length=1):
    if isinstance(t, tuple):
        return t
    return ((t,) * length)

def identity(t, *args, **kwargs):
    return t

# small helper modules

def Upsample(dim, dim_out=None):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="nearest")
        nn.Conv2d(dim, default(dim_out, dim), 3, padding=1)
    )

def Downsample(dim, dim_out=False):
    return nn.Sequential(
        Rearrange('b c (h p1) (w p2) -> b (c p1 p2) h w', p1 = 2, p2 = 2),
        nn.Conv2d(dim * 4, default=(dim_out, dim), 1)
    )

class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1,1))
    def forward(self ,x):
        return F.normalize(x, dim=1) * self.g * (x.shape[1] ** 0.5)
    

# class RMSNorm(nn.Module):
#     def __init__(self, dim: int, eps: float = 1e-6):
#         super().__init__()
#         self.eps = eps
#         self.weight = nn.Parameter(torch.ones(1, dim, 1, 1))  # Learnable per-channel scaling

#     def _norm(self, x: torch.Tensor):
#         # Compute RMS normalization over spatial dimensions (height and width)
#         mean_square = x.pow(2).mean(dim=[2, 3], keepdim=True)  # Mean over height & width
#         return x * torch.rsqrt(mean_square + self.eps)  # Normalize

#     def forward(self, x):
#         # Apply learnable scaling and normalization
#         return self.weight * self._norm(x)


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self, x):
        device = x.device
        # Original formula from "Attention is all you need": PE(pos, i) = func(pos / 10000^(i/dim))
        # The following positional encodings are evaluated in log space.
        # Input: (B). Output: (B, dim)
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:,None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb
    
# building block modules
# Represents a convolution block with a group normalization layer and an activation function
class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.proj =nn.Conv2d(dim, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        x=self.proj(x)
        x=self.norm(x)
        if  exists(scale_shift):
            scale, shift = scale_shift
            x =x*(scale+1) +shift
        x = self.act(x)
        return x
    
# Each ResnetBlock is composed of two blocks and a residual connection
class ResnetBlock(nn.Module):
    def __init__(self, dim,dim_out, *,  time_emb_dim=None, groups=8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_cov = nn.Conv2d(dim, dim_out,1) if dim!=dim_out else nn.Identity()
    

    def forward(self, x, time_emb=None):
        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            # add two dimensions to time_emb
            time_emb = rearrange(time_emb, 'b c -> b c 1 1')
            # Divide the time embedding into two parts along the channel dimension
            scale_shift = time_emb.chuck(2, dim=1)
        h = self.block1(x, scale_shift=scale_shift)
        h = self.block2(h)
        return h + self.res_cov(x)
    

class Attend(nn.Module):
    def __init__(self, dropput=0.):
        super().__init__()
        self.dropout = dropput
        self.attn_dropout =  nn.Dropout(dropput)
    def forward(self, q,k,v):
        q_len, k_len, device = q.shape[-2],k.shape[-2], q.device
        scale = q.shape[-1] ** 0.5
         # similarity
        sim = torch.matmul(q, k.transpose(-2,-1)) * scale
         # attention
        attn = sim.softmax(dim=-1)

        attn =self.attn_dropout(attn)

        out = torch.matmul(attn, v)
        return out
        

class Attention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head=32):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads