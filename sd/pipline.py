import torch
import numpy as np
from tqdm import tqdm
from ddpm import DDPMSampler

WIDTH = 512
HEIGHT = 512
LATENTS_WIDTH = WIDTH // 8
LATENTS_HEIGHT = HEIGHT // 8

def generate(prompt:str, uncond_prompt:str, input_image = None, 
             strength=0.8, do_cfg=True, cfg_scale=7.5, sampler_name="ddpm", 
             n_inference_steps=50, models={}, seed=None):
