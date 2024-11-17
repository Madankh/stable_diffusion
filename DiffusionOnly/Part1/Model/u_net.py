import torch
import torch.nn as nn
from utils import default
from Model.composants import Upsample, DownSample, ConvNextBlock, SinusoiidalPosEmb, BlockAttention

class OneResUNet(nn.Module):
    