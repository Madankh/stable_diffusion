import torch
from torch import nn
from torch.nn import functional as F
from decoder import VAE_AttentionBlock, VAE_ResidualBlock

class VAE_Encoder(nn.Sequential):
    def __init__(self):
        super().__init__(
        # (Batch, channels, Height, Width) --? (Batch_size , 128 , Height, Width)
        nn.Conv2d(3,128, kernel_size=3, stride=1,padding=1),
        # (Batch , 128 , Height, Width) ---> (Batch_size , 128 , Height, Width)
        VAE_ResidualBlock(128 , 128),
        # (Batch , 128 , Height, Width) ---> (Batch_size , 128 , Height, Width)
        VAE_ResidualBlock(128 , 128),

        # (Batch_size , 128, Height, Width) ----> (Batch_size , 128, Height/2, Width/2)
        nn.Conv2d(128,128, kernel_size=3, stride=2, padding=0),
        # (Batch_size , 128, Height/2, Width/2) --> (Batch_size , 256, Height/2, Width/2)
        VAE_ResidualBlock(128 , 256),
        # (Batch_size , 256, Height/2, Width/2) --> (Batch_size , 256, Height/2, Width/2)
        VAE_ResidualBlock(256 , 256),
         # (Batch_size , 256, Height/2, Width/2) --> (Batch_size , 256, Height/4, Width/4)
         nn.Conv2d(256,256, kernel_size=3, stride=2, padding=0),
         VAE_ResidualBlock(256 , 512),
    )
        