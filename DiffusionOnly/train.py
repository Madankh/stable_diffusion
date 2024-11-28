import torch
from data import DiffSet
from model import DiffusionModel
from torch.utils.data import DataLoader
import imageio
import glob
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os

def sample_gif(model, train_dataset, output_dir) -> None:
    gif_shape = [3,3] # The gif will be a grid of images of this shape
    sample_batch_size = gif_shape[0] * gif_shape[1]
    n_hold_final = 100  # How many samples to append to the end of the GIF to hold the final image fixed

    # Generate samples from denoising process
    gen_samples = []
    sampled_steps = []
    # Generate random noise
    x = torch.randn(
        (sample_batch_size, train_dataset.depth, train_dataset.size, train_dataset.size)
    )
    sample_steps = torch.arange(model.t_range - 1, 0, -1)
    sampled_t = 0
    # Denoise the initial noise for T steps
    for t in tqdm(sampled_steps, desc="Sampling"):
        x = model.denoise_sample(x, t)
        sampled_t = t
        gen_samples.append(x)
        sample_steps.append(sampled_t)
    # add the final image to the end of the GIF many times to hold it fixed
    for _ in range(n_hold_final):
        gen_samples.append(x)
        sample_steps.append(sampled_t)
    gen_samples = torch.stack(gen_samples, dim=0).moveaxis(2,4).squeeze(-1)
    gen_samples = (gen_samples.clamp(-1,1)+1)/2

    gen_samples = (gen_samples * 255).type(torch.uint8)

    get_samples = gen_samples.reshape(
        -1, 
        gif_shape[0],
        gif_shape[1],
        train_dataset.size, 
        train_dataset.size,
        train_dataset.depth,
    )
