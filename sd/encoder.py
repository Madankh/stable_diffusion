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
         # (Batch_size , 512, Height/4, Width/4) --> (Batch_size , 512, Height/4, Width/4)
         VAE_ResidualBlock(512, 512),
         # (Batch_size , 512, Height/4, Width/4) --> (Batch_size , 512, Height/8, Width/8)
         nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=0),
         VAE_ResidualBlock(512, 512),

         VAE_ResidualBlock(512, 512),
         
         VAE_ResidualBlock(512, 512),

         # (Batch_Size, 512, Height/8, Width / B) -> (Batch_size, 512, Height/8, Width / 8)
         VAE_AttentionBlock(512),
         # (Batch_Size, 512, Height/8, Width / B) -> (Batch_size, 512, Height/8, Width / 8)
         VAE_ResidualBlock(512, 512),

         nn.GroupNorm(32, 512),
         nn.SiLU(),

         # (Batch_Size, 512, Height/8, Width / B) -> (Batch_size, 512, Height/8, Width / 8)
         nn.Conv2d(512, 8, kernel_size = 3, padding=1),
         nn.Conv2d(8, 8, kernel_size=1, padding=0)

    )
        
    def forward(self, x:torch.Tensor, noise:torch.Tensor) -> torch.Tensor:
        # x : (Batch_size, Channel, Height, width)
        # noise : (Batch_size, out_channels, Height/8, width/8)
        for module in self:
            if getattr(module, "stride", None) == (2,2):
                # (Padding left, padding right, padding_top, paddng_bottom)
                x = F.pad(x, (0 , 1 , 0 , 1))
            x = module(x)

        # Batch_size , 8, Height/8, Width/8 -> two tensors of shape (B, 4 , H/8 , W/8)
        mean, log_variance = torch.chunk(x, 2, dim=1)
        # Clamp the log variance between -30 and 20, so that the variance is between (circa) 1e-14 and 1e8. 
        # (Batch_Size, 4, Height / 8, Width / 8) -> (Batch_Size, 4, Height / 8, Width / 8)
        log_variance = torch.clamp(log_variance, -30, 20)
        # (Batch_Size, 4, Height / 8, Width / 8) -> (Batch_Size, 4, Height / 8, Width / 8)
        variance = log_variance.exp()
        # (Batch_Size, 4, Height / 8, Width / 8) -> (Batch_Size, 4, Height / 8, Width / 8)
        stdv = variance.sqrt()
        x = mean + stdv * noise


        x*=0.18215
        return x