import torch
import torch.nn as nn
import pytorch_lightning as pl
import math
from modules import *

class DiffusionModel(nn.Module):
    def __init__(self,in_size, t_range, img_depth):
        super().__init__()
        self.beta_small = 1e-4
        self.beta_large = 0.02
        self.t_range = t_range
        self.in_size = in_size

        self.unet = Unet(dim=64, dim_mults=(1,2,4,8), channels=img_depth)

    def forward(self,x,t):
        return self.unet(x,t)
    
    def beta(self, t):
        # Just a simple linear interpolation between beta_small and beta_large based on t
        return self.beta_small + (t / self.t_range) * (self.beta_large - self.beta_small)
    
    def alpha(self, t):
        return 1 - self.beta(t)
    
    def alpha_bar(self,t):
        return math.prod([self.alpha(j) for j in range(t)])
    
    