import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import numpy as np
from torchvision.utils import save_image

from models.generator import Generator
from models.discriminator import Discriminator
from utils.data_loader import get_dataloader

# --- Advanced Hyperparameters based on the new advice ---
LEARNING_RATE_G = 0.0002
LEARNING_RATE_D = 0.00005  # lower discriminator learning rate for stability
BETA1 = 0.0  # WGAN often uses Beta1=0.0
BETA2 = 0.9
NUM_EPOCHS = 50
LATENT_DIM = 100
BATCH_SIZE = 64
IMAGE_CHANNELS = 3

# Training regime parameters
NUM_D_STEPS = 5  # WGAN recommends training D more frequently
NUM_G_STEPS = 1

# Regularization parameters
GRADIENT_PENALTY_WEIGHT = 10.0  # standard for WGAN-GP
NOISE_STD = 0.0  # not using noise with WGAN-GP

# Loss function choice: 'wgan' is our choice for stability
LOSS_TYPE = 'wgan'

# --- Directory setup ---
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(project_root, 'output')
os.makedirs(os.path.join(OUTPUT_DIR, 'generated_images'), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, 'checkpoints'), exist_ok=True)


def add_noise_to_inputs(real_images, fake_images, noise_std=NOISE_STD):
    # we're not using this with WGAN-GP, but it's a good helper to keep
    if noise_std > 0:
        real_noise = torch.randn_like(real_images) * noise_std
        fake_noise = torch.randn_like(fake_images) * noise_std
        real_images = real_images + real_noise
        fake_images = fake_images + fake_noise
        real_images = torch.clamp(real_images, -1, 1)
        fake_images = torch.clamp(fake_images, -1, 1)
    return real_images, fake_images


def gradient_penalty(discriminator, real_images, fake_images, device):
    # this calculates the gradient penalty, a key part of WGAN-GP
    batch_size = real_images.size(0)
    alpha = torch.rand(batch_size, 1, 1, 1, device=device)
    
    interpolated = alpha * real_images + (1 - alpha) * fake_images
    interpolated.requires_grad_(True)
    
    normal_out, forensic_out = discriminator(interpolated)
    combined_out = (normal_out + forensic_out) / 2
    
    gradients = torch.autograd.grad(
        outputs=combined_out,
        inputs=interpolated,
        grad_outputs=torch.ones_like(combined_out, device=device),
        create_graph=True,
        retain_graph=True
    )[0]
    
    gradient_norm = gradients.view(batch_size, -1).norm(2, dim=1)
    penalty = torch.mean((gradient_norm - 1) ** 2)
    return penalty


def compute_discriminator_loss(discriminator, real_images, fake_images, device):
    # compute the wgan-gp loss for the discriminator
    normal_out_real, forensic_out_real = discriminator(real_images)
    normal_out_fake, forensic_out_fake = discriminator(fake_images)
    
    real_score = (normal_out_real + forensic_out_real) / 2
    fake_score = (normal_out_fake + forensic_out_fake) / 2
    
    loss_wgan = -torch.mean(real_score) + torch.mean(fake_score)
    
    gp = gradient_penalty(discriminator, real_images, fake_images.detach(), device)
    
    return loss_wgan + GRADIENT_PENALTY_WEIGHT * gp


def compute_generator_loss(generator, discriminator, noise, device):
    # compute the wgan-gp loss for the generator
    fake_images = generator(noise)
    normal_out, forensic_out = discriminator(fake_images)
    
    fake_score = (normal_out + forensic_out) / 2
    
    return -torch.mean(fake_score)


def train():
    # --- Device Configuration ---
    if torch.backends.cuda.is_available():
        device = torch.device("cuda")
    elif torch.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    print(f"Using device: {device}")

    # Initialize models
    generator = Generator().to(device)
    discriminator = Discriminator(use_spectral_norm=True).to(device)
    print("Models initialized with improved architectures.")

    # Load data
    data_dir = os.path.join(project_root, 'data', 'processed', 'resized')
    dataloader = get_dataloader(data_dir)
    print("Data loader created.")

    # Optimizers with different learning rates
    g_optimizer = optim.Adam(generator.parameters(), lr=LEARNING_RATE_G, betas=(BETA1, BETA2))
    d_optimizer = optim.Adam(discriminator.parameters(), lr=LEARNING_RATE_D, betas=(BETA1, BETA2))
    
    # WGAN doesn't use a criterion (BCELoss), so we set it to None
    criterion = None
    
    # Fixed noise for consistent image generation
    fixed_noise = torch.randn(64, LATENT_DIM, 1, 1, device=device)

    print(f"\nAdvanced Training Configuration:")
    print(f"Loss type: {LOSS_TYPE}")
    print(f"G_LR: {LEARNING_RATE_G}, D_LR: {LEARNING_RATE_D} (ratio: {LEARNING_RATE_G/LEARNING_RATE_D:.1f})")
    print(f"D_steps: {NUM_D_STEPS}, G_steps: {NUM_G_STEPS}")
    print(f"Gradient Penalty Weight: {GRADIENT_PENALTY_WEIGHT}")
    
    # --- Main Training Loop ---
    print(f"\nStarting main training loop for {NUM_EPOCHS} epochs...")
    
    for epoch in range(NUM_EPOCHS):
        total_loss_d = 0.0
        total_loss_g = 0.0
        
        for i, (real_images, _) in enumerate(dataloader):
            real_images = real_images.to(device)
            batch_size = real_images.size(0)
            
            # --- Train Discriminator (more frequently) ---
            for _ in range(NUM_D_STEPS):
                d_optimizer.zero_grad()
                
                # Generate a batch of fake images
                noise = torch.randn(batch_size, LATENT_DIM, 1, 1, device=device)
                fake_images = generator(noise).detach()
                
                loss_d = compute_discriminator_loss(discriminator, real_images, fake_images, device)
                
                loss_d.backward()
                d_optimizer.step()
                total_loss_d += loss_d.item()
            
            # --- Train Generator ---
            for _ in range(NUM_G_STEPS):
                g_optimizer.zero_grad()
                noise = torch.randn(batch_size, LATENT_DIM, 1, 1, device=device)
                
                loss_g = compute_generator_loss(generator, discriminator, noise, device)
                
                loss_g.backward()
                g_optimizer.step()
                total_loss_g += loss_g.item()

            if i % 10 == 0 or i == len(dataloader) - 1:
                print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Batch [{i+1}/{len(dataloader)}] "
                      f"Loss_D: {loss_d.item():.4f}, Loss_G: {loss_g.item():.4f}")

        # --- Epoch Summary ---
        avg_loss_d = total_loss_d / (len(dataloader) * NUM_D_STEPS)
        avg_loss_g = total_loss_g / (len(dataloader) * NUM_G_STEPS)
        print(f"=== Epoch [{epoch+1}/{NUM_EPOCHS}] Summary ===")
        print(f"Avg Loss_D: {avg_loss_d:.4f}, Avg Loss_G: {avg_loss_g:.4f}")
        
        # --- Save generated images from fixed noise ---
        with torch.no_grad():
            generator.eval()
            fake_samples = generator(fixed_noise).detach().cpu()
            save_image(fake_samples, os.path.join(OUTPUT_DIR, 'generated_images', f'epoch_{epoch+1}.png'), normalize=True, nrow=8)
            generator.train()

        # --- Save checkpoints ---
        if (epoch + 1) % 5 == 0:
            torch.save({
                'generator_state_dict': generator.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'g_optimizer_state_dict': g_optimizer.state_dict(),
                'd_optimizer_state_dict': d_optimizer.state_dict(),
                'epoch': epoch + 1,
            }, os.path.join(OUTPUT_DIR, 'checkpoints', f'gan_checkpoint_epoch_{epoch+1}.pth'))
            print(f"Checkpoint saved for epoch {epoch+1}")

    print("\nTraining completed successfully!")


if __name__ == '__main__':
    train()
