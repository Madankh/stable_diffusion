import torch
import torch.nn as nn
import numpy

class DDPMSampler:
    def __init__(self, generator:torch.Generator, num_training_steps=1000, beta_start:float=0.00085, beta_end:float=0.0120):
        
        pass