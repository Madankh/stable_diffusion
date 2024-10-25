import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention, CrossAttention

class TimeEmbedding(nn.Module):
    def __init__(self, n_embd:int):
        super().__init__()
        self.linear_1 = nn.Linear(n_embd, 4*n_embd)
        self.linear_2 = nn.Linear(n_embd*4, 4*n_embd)

    def forward(self,x:torch.Tensor)->torch.Tensor:
        # x
        x = self.linear_1(x)
        x = F.silu(x)
        x = self.linear_2(x)

        return x
    
class SwitchSequential(nn.Sequential):
    def forward(self, x:torch.Tensor, context:torch.Tensor, time:torch.Tensor):
        for layer in self:
            if isinstance(layer, UNET_AttentionBlock):
                x = layer(x, context)
            elif isinstance(layer, UNET_ResidualBlock):
                x = layer(x, time)
            else:
                x = layer(x)

        return x

class Upsample(nn.Module):
    def __init__(self, channels:int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
    
    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


class UNET(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoders = nn.Module([
            # (Batch_Size, 4, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8)
            SwitchSequential(nn.Conv2d(4 , 320, kernel_size=3, padding=1)),

            # (Batch_Size, 320, Height / 8, Width / 8) -> # (Batch_Size, 320, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8)
            SwitchSequential(UNET_ResidualBlock(320 , 320), UNET_AttentionBlock(8,40)),

            SwitchSequential(UNET_ResidualBlock(320 , 320), UNET_AttentionBlock(8,40))

            SwitchSequential(nn.Conv2d(320, 320, kernel_size=3, stride=2, padding=1)),

            SwitchSequential(UNET_ResidualBlock(320 , 640), UNET_AttentionBlock(8,80)),

            SwitchSequential(UNET_ResidualBlock(640 , 640), UNET_AttentionBlock(8,80)),

            SwitchSequential(nn.Conv2d(640, 640, kernel_size=3, stride=2, padding=1)),

            SwitchSequential(UNET_ResidualBlock(640,1280), UNET_AttentionBlock(8, 160)),

            SwitchSequential(UNET_ResidualBlock(1280,1280), UNET_AttentionBlock(8, 160))

            SwitchSequential(nn.Conv2d(1280,1280, kernel_size=3, stride=2, padding=1)),

            SwitchSequential(UNET_ResidualBlock(1280, 1280)),

            SwitchSequential(UNET_ResidualBlock(1280, 1280)),
            
        ])

        self.bottlenek = SwitchSequential(
            UNET_ResidualBlock(1280, 1280),

            UNET_AttentionBlock(8, 160).

            UNET_ResidualBlock(1280, 1280),

        )

        self.decoders = nn.Sequential([
            SwitchSequential(UNET_ResidualBlock(2560, 1280)),

            SwitchSequential(UNET_ResidualBlock(2560, 1280)),

            SwitchSequential(UNET_ResidualBlock(2560, 1280), Upsample(1280)),

            # (Batch_Size, 2560, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 32, Width / 32)
            SwitchSequential(UNET_ResidualBlock(2560, 1280), UNET_AttentionBlock(8, 160)),
            
            # (Batch_Size, 2560, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 32, Width / 32)
            SwitchSequential(UNET_ResidualBlock(2560, 1280), UNET_AttentionBlock(8, 160)),

            SwitchSequential(UNET_ResidualBlock(1920, 1280), UNET_AttentionBlock(8, 160), Upsample(1280)),

            SwitchSequential(UNET_ResidualBlock(1920 , 640), UNET_AttentionBlock(8, 80)),

            SwitchSequential(UNET_ResidualBlock(1280 , 640), UNET_AttentionBlock(8, 80)),

            SwitchSequential(UNET_ResidualBlock(960 , 640), UNET_AttentionBlock(8, 80), Upsample(640)),

            SwitchSequential(UNET_ResidualBlock(960 , 320), UNET_AttentionBlock(8, 40)),

            SwitchSequential(UNET_ResidualBlock(640, 320), UNET_AttentionBlock(8,40)),

            SwitchSequential(UNET_ResidualBlock(640, 320), UNET_AttentionBlock(8,40)),

        ])

class UNET_OutputLayer(nn.Module):
    def __init__(self, in_channels:int, out_channels:int):
        super().__init__()
        self.groupnorm = nn.GroupNorm(32, in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.groupnorm(x)
        x = F.silu(x)
        x = self.conv(x)
    
        return x

class Diffusion(nn.Module):

    def __init__(self):
        self.time_embedding = TimeEmbedding(320)
        self.unet = UNET()
        self.final = UNET_OutputLayer(320, 4)

    def forward(self, latent:torch.Tensor, context:torch.Tensor, time:torch.Tensor):
        # Latent : (Batch_size, 4, Height/8, Width/8)
        # context : (Batch_size, seq_len, dim)
        # time : (1,320)
        time = self.time_embedding(time)

        # (Batch_size, 4, hEIGHT/8, Width/8) --> # (Batch_size, 320, hEIGHT/8, Width/8)
        output = self.unet(latent, context, time)

        output = self.final(output)

        return output