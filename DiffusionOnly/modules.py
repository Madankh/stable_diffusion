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
    def __init__(
        self,
        dim,
        heads = 4,
        dim_head = 32
    ):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads

        self.norm = RMSNorm(dim)
        self.attend = Attend()

        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h (x y) c', h = self.heads), qkv)
        out = self.attend(q, k, v)
        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x = h, y = w)
        return self.to_out(out)
    
class Unet(nn.Module):
    def __init__(
            self,
            dim,
            init_dim=None,
            out_dim=None,
            dim_mults=(1,2,4,8),
            channels=3,
            resnet_block_groups=8
            ):
        super().__init__()

        # determine dimensions
        self.channels = channels # input channels 
        input_channels = channels

        init_dim = default(init_dim, dim)  #Initial dimension of the featuere map extracted from the input image
        self.init_conv = nn.Conv2d(input_channels, init_dim, 7, padding=3)

        dims =  [init_dim,  *map(lambda m:dim * m, dim_mults)] # Builds multiples of the initial dimensions, used for the encoder and the decoder
        in_out = list(zip(dims[:-1],dim[1:])) # Builds a map of input-output dimensions for the encoder and the decoder
        
        block_klass = partial(ResnetBlock, groups=resnet_block_groups) # Initializes the number of groups for Group Normalization

        # time embeddings
        time_dim = dim*4 #  the size of the time embedding
        sinu_pos_emb = SinusoidalPosEmb(dim) # Positional encodings just like in normal transfomer
        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        # layerss
        self.downs  = nn.modules([])
        self.ups = nn.modules([])
        num_resolutions = len(in_out) # number of layers (up and downs) in the unet

        for ind, ((dim_in, dim_out)) in enumerate(in_out):
            is_last  = ind >= (num_resolutions -  1)

            self.downs.append(nn.ModuleList([
                block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                Attention(dim_in),
                Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out,3, padding=1)
            ]))

        mid_dim = dims[:-1]  # The dimension of the middle layer of the UNet = 512
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = Attention(mid_dim)
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)

        for ind, ((dim_in, dim_out)) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)

            self.ups.append(nn.ModuleList([
                block_klass(dim_out + dim_in, dim_out, time_emb_dim = time_dim),
                block_klass(dim_out + dim_in, dim_out, time_emb_dim = time_dim),
                Attention(dim_out),
                Upsample(dim_out, dim_in) if not is_last else  nn.Conv2d(dim_out, dim_in, 3, padding = 1)
            ]))
        default_out_dim = channels
        self.out_dim = default(out_dim, default_out_dim) # channels of the output image

        self.final_res_block = block_klass(dim*2, dim, time_emb_dim=time_dim)
        self.final_conv =  nn.Conv2d(dim,  self.out_dim, 1)

    def forward(self, x,  time):
        x = self.init_conv(x)
        r = x.clone()
        t = self.time_mlp(time)

        h = []

        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            h.append(x)
            x = block2(x, t)
            x = attn(x)
            h.append(x)
            x = downsample(x)
        
        x = self.mid_block1(x, t)
        x  = self.mid_attn(x) + x
        x = self.mid_block2(x,  t)

        for block1, block2, attn, downsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)
            x = torch.cat((x, h.pop()), dim=1)
            x = block2(x, t)
            x = attn(x) + x
            x = Upsample(x)

        x = torch.cat((x,r), dim=1)
        x = self.final_res_block(x, t)
        return self.final_conv(x)