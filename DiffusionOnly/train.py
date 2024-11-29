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

    # Add a text to the first image in each grid to indicate the step shown
    def add_text_to_image(image, text):
        black_image = np.zeros_like(image.numpy())
        black_image = Image.fromarray(black_image, "RGB")
        draw = ImageDraw.Draw(black_image)
        font = ImageFont.load_default()
        draw.text((0, 0), text, (255, 255, 255), font=font)
        black_image = torch.tensor(np.array(black_image))
        return black_image

    for i in range(gen_samples.shape[0]):
        gen_samples[i, 0, 0] = add_text_to_image(
            gen_samples[i, 0, 0], f"{sampled_steps[i]}"
        )

    def stack_samples(gen_samples, stack_dim):
        gen_samples = list(torch.split(gen_samples, 1, dim=1))
        for i in range(len(gen_samples)):
            gen_samples[i] = gen_samples[i].squeeze(1)
        return torch.cat(gen_samples, dim=stack_dim)

    gen_samples = stack_samples(gen_samples, 2)
    gen_samples = stack_samples(gen_samples, 2)

    output_file = f"{output_dir}/pred.gif"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    imageio.mimsave(
        output_file, list(gen_samples.squeeze(-1)), format="GIF", duration=20
    )

def train_model(config: dict):
    # Load the dataset
    train_dataset = DiffSet(True, config["dataset"])
    val_dataset = DiffSet(False, config["dataset"])
    
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=4)
    
    # Initialize the model
    model = DiffusionModel(
        train_dataset.size * train_dataset.size,
        config["diffusion_steps"],
        train_dataset.depth,
    )
    
    if config["load_model"]:
        last_checkpoint = glob.glob(
            f"./checkpoints/{config['dataset']}/version_{config['load_version_num']}/*.pth"
        )[-1]
        print(f"Loading model from {last_checkpoint}")
        model.load_state_dict(torch.load(last_checkpoint))
    
    # Define optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = torch.nn.MSELoss()  # Example loss function (depends on your task)
    
    # Prepare for GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Training loop
    num_epochs = config["max_epoch"]
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            # Move data to GPU
            x, y = batch[0].to(device), batch[1].to(device)  # Assuming x=input, y=target
            
            # Forward pass
            outputs = model(x)
            
            # Compute loss
            loss = loss_fn(outputs, y)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}] - Training Loss: {train_loss:.4f}")
        
        # Validation loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                x, y = batch[0].to(device), batch[1].to(device)
                outputs = model(x)
                loss = loss_fn(outputs, y)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}] - Validation Loss: {val_loss:.4f}")
        
        # Save checkpoint
        os.makedirs(f"./checkpoints/{config['dataset']}/", exist_ok=True)
        torch.save(model.state_dict(), f"./checkpoints/{config['dataset']}/epoch_{epoch+1}.pth")
    
    return model, train_dataset, f"./checkpoints/{config['dataset']}/"
