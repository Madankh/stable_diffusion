import torch
from torch import nn
from torch.nn import functional as F
import math

class SelfAttention(nn.Module):
    def __init__(self, n_heads:int , d_embed:int, in_proj_bias=True, out_proj_bias=True):

        super().__init__()
        self.in_proj = nn.Linear(d_embed, 3*d_embed, bias=in_proj_bias)
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads

    def forward(self, x:torch.Tensor, causal_mask=False):
        # x : (Batch_size, seq_len, Dim)
        input_shape = x.shape
        batch_size , sequence_length, d_embed = input_shape