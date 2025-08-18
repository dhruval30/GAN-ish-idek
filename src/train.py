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

# --- Advanced Hyperparameters Based on Community Best Practices ---
LEARNING_RATE_G = 0.0002
LEARNING_RATE_D = 0.00006  # 0.3× of generator (community recommendation)
BETA1 = 0.0  # Changed from 0.5 to 0.0 for better stability
BETA2 = 0.999
NUM_EPOCHS = 50
LATENT_DIM = 100
BATCH_SIZE = 64
IMAGE_CHANNELS = 3

# Training regime parameters
GENERATOR_PRETRAIN_EPOCHS = 2  # Give generator a head start
D_THRESHOLD = 0.8  # Only train discriminator if its accuracy is below this
G_STEPS_PER_D = 1  # Initially balanced, will be adjusted dynamically
MAX_G_STEPS = 3    # Maximum generator steps per discriminator step

# Regularization parameters
LABEL_SMOOTH_REAL = 0.9
LABEL_SMOOTH_FAKE = 0.0  # One-sided label smoothing
NOISE_STD = 0.05  # Standard deviation for input noise
GRADIENT_PENALTY_WEIGHT = 10.0  # For WGAN-GP style training

# Loss function choice: 'bce', 'wgan', 'hinge'
LOSS_TYPE = 'bce'  # Start with BCE, can switch to 'wgan' for more stability

# --- Directory setup ---
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(project_root, 'output')
os.makedirs(os.path.join(OUTPUT_DIR, 'generated_images'), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, 'checkpoints'), exist_ok=True)


def add_noise_to_inputs(real_images, fake_images, noise_std=NOISE_STD):
    """Add small noise to inputs to make discriminator's job harder"""
    if noise_std > 0:
        real_noise = torch.randn_like(real_images) * noise_std
        fake_noise = torch.randn_like(fake_images) * noise_std
        real_images = real_images + real_noise
        fake_images = fake_images + fake_noise
        # Clamp to valid range
        real_images = torch.clamp(real_images, -1, 1)
        fake_images = torch.clamp(fake_images, -1, 1)
    return real_images, fake_images


def gradient_penalty(discriminator, real_images, fake_images, device):
    """Compute gradient penalty for WGAN-GP"""
    batch_size = real_images.size(0)
    alpha = torch.rand(batch_size, 1, 1, 1, device=device)
    
    interpolated = alpha * real_images + (1 - alpha) * fake_images
    interpolated.requires_grad_(True)
    
    normal_out, forensic_out = discriminator(interpolated)
    # Combine both outputs for gradient penalty
    combined_out = (normal_out + forensic_out) / 2
    
    gradients = torch.autograd.grad(
        outputs=combined_out,
        inputs=interpolated,
        grad_outputs=torch.ones_like(combined_out),
        create_graph=True,
        retain_graph=True
    )[0]
    
    gradient_norm = gradients.view(batch_size, -1).norm(2, dim=1)
    penalty = torch.mean((gradient_norm - 1) ** 2)
    return penalty


def compute_discriminator_loss(discriminator, real_images, fake_images, criterion, device, loss_type='bce'):
    """Compute discriminator loss based on specified loss type"""
    batch_size = real_images.size(0)
    
    if loss_type == 'bce':
        # Standard BCE loss with label smoothing
        real_labels = torch.full((batch_size,), LABEL_SMOOTH_REAL, dtype=torch.float, device=device)
        fake_labels = torch.full((batch_size,), LABEL_SMOOTH_FAKE, dtype=torch.float, device=device)
        
        normal_out_real, forensic_out_real = discriminator(real_images)
        normal_out_fake, forensic_out_fake = discriminator(fake_images)
        
        loss_real = criterion(normal_out_real, real_labels) + criterion(forensic_out_real, real_labels)
        loss_fake = criterion(normal_out_fake, fake_labels) + criterion(forensic_out_fake, fake_labels)
        
        return (loss_real + loss_fake) / 2
    
    elif loss_type == 'wgan':
        # Wasserstein loss
        normal_out_real, forensic_out_real = discriminator(real_images)
        normal_out_fake, forensic_out_fake = discriminator(fake_images)
        
        # WGAN loss: maximize D(real) - D(fake)
        real_score = (normal_out_real + forensic_out_real) / 2
        fake_score = (normal_out_fake + forensic_out_fake) / 2
        
        return -torch.mean(real_score) + torch.mean(fake_score)
    
    elif loss_type == 'hinge':
        # Hinge loss
        normal_out_real, forensic_out_real = discriminator(real_images)
        normal_out_fake, forensic_out_fake = discriminator(fake_images)
        
        real_score = (normal_out_real + forensic_out_real) / 2
        fake_score = (normal_out_fake + forensic_out_fake) / 2
        
        loss_real = torch.mean(F.relu(1.0 - real_score))
        loss_fake = torch.mean(F.relu(1.0 + fake_score))
        
        return loss_real + loss_fake


def compute_generator_loss(generator, discriminator, noise, criterion, device, loss_type='bce'):
    """Compute generator loss based on specified loss type"""
    fake_images = generator(noise)
    normal_out, forensic_out = discriminator(fake_images)
    batch_size = noise.size(0)
    
    if loss_type == 'bce':
        # Generator wants discriminator to think fakes are real
        real_labels = torch.ones(batch_size, dtype=torch.float, device=device)
        return criterion(normal_out, real_labels) + criterion(forensic_out, real_labels)
    
    elif loss_type == 'wgan':
        # WGAN generator loss: maximize D(fake)
        fake_score = (normal_out + forensic_out) / 2
        return -torch.mean(fake_score)
    
    elif loss_type == 'hinge':
        # Hinge generator loss
        fake_score = (normal_out + forensic_out) / 2
        return -torch.mean(fake_score)


def train():
    # --- Device Configuration ---
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
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
    
    # Loss function
    if LOSS_TYPE == 'bce':
        criterion = nn.BCELoss()
    else:
        criterion = None  # WGAN and Hinge don't use BCE
    
    # Fixed noise for consistent image generation
    fixed_noise = torch.randn(64, LATENT_DIM, 1, 1, device=device)

    print(f"\nAdvanced Training Configuration:")
    print(f"Loss type: {LOSS_TYPE}")
    print(f"G_LR: {LEARNING_RATE_G}, D_LR: {LEARNING_RATE_D} (ratio: {LEARNING_RATE_G/LEARNING_RATE_D:.1f})")
    print(f"Generator pretraining: {GENERATOR_PRETRAIN_EPOCHS} epochs")
    print(f"Dynamic training with D_threshold: {D_THRESHOLD}")
    print(f"Input noise std: {NOISE_STD}")
    
    # --- Generator Pretraining ---
    if GENERATOR_PRETRAIN_EPOCHS > 0:
        print(f"\nPretraining generator for {GENERATOR_PRETRAIN_EPOCHS} epochs...")
        generator.train()
        discriminator.eval()  # Keep discriminator frozen
        
        for epoch in range(GENERATOR_PRETRAIN_EPOCHS):
            for i, (real_images, _) in enumerate(dataloader):
                g_optimizer.zero_grad()
                batch_size = real_images.size(0)
                noise = torch.randn(batch_size, LATENT_DIM, 1, 1, device=device)
                
                loss_g = compute_generator_loss(generator, discriminator, noise, criterion, device, LOSS_TYPE)
                loss_g.backward()
                g_optimizer.step()
                
                if i % 20 == 0:
                    print(f"Pretrain Epoch [{epoch+1}/{GENERATOR_PRETRAIN_EPOCHS}], Batch [{i+1}/{len(dataloader)}], G_Loss: {loss_g.item():.4f}")

    # --- Main Training Loop ---
    print(f"\nStarting main training loop for {NUM_EPOCHS} epochs...")
    
    for epoch in range(NUM_EPOCHS):
        total_loss_d = 0.0
        total_loss_g = 0.0
        d_updates = 0
        g_updates = 0
        current_g_steps = G_STEPS_PER_D

        for i, (real_images, _) in enumerate(dataloader):
            real_images = real_images.to(device)
            batch_size = real_images.size(0)
            
            # Generate fake images
            noise = torch.randn(batch_size, LATENT_DIM, 1, 1, device=device)
            with torch.no_grad():
                fake_images = generator(noise)
            
            # Add noise to inputs
            real_images_noisy, fake_images_noisy = add_noise_to_inputs(real_images, fake_images.detach())
            
            # --- Discriminator Training (with threshold) ---
            # Calculate discriminator accuracy to decide if we should train it
            with torch.no_grad():
                normal_out_real, forensic_out_real = discriminator(real_images_noisy)
                normal_out_fake, forensic_out_fake = discriminator(fake_images_noisy)
                
                real_acc = ((normal_out_real > 0.5).float().mean() + (forensic_out_real > 0.5).float().mean()) / 2
                fake_acc = ((normal_out_fake < 0.5).float().mean() + (forensic_out_fake < 0.5).float().mean()) / 2
                d_acc = (real_acc + fake_acc) / 2
            
            # Only train discriminator if its accuracy is below threshold
            if d_acc < D_THRESHOLD:
                d_optimizer.zero_grad()
                
                loss_d = compute_discriminator_loss(discriminator, real_images_noisy, fake_images_noisy.detach(), criterion, device, LOSS_TYPE)
                
                # Add gradient penalty for WGAN
                if LOSS_TYPE == 'wgan':
                    gp = gradient_penalty(discriminator, real_images, fake_images.detach(), device)
                    loss_d += GRADIENT_PENALTY_WEIGHT * gp
                
                loss_d.backward()
                d_optimizer.step()
                total_loss_d += loss_d.item()
                d_updates += 1
                
                # If discriminator is too weak, train it more
                if d_acc < 0.6:
                    current_g_steps = max(1, current_g_steps - 1)
                else:
                    current_g_steps = min(MAX_G_STEPS, current_g_steps + 1)
            
            # --- Generator Training (multiple steps) ---
            for g_step in range(current_g_steps):
                g_optimizer.zero_grad()
                noise = torch.randn(batch_size, LATENT_DIM, 1, 1, device=device)
                
                loss_g = compute_generator_loss(generator, discriminator, noise, criterion, device, LOSS_TYPE)
                loss_g.backward()
                g_optimizer.step()
                total_loss_g += loss_g.item()
                g_updates += 1

            # Logging
            if i % 15 == 0 or i == len(dataloader) - 1:
                print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Batch [{i+1}/{len(dataloader)}] "
                      f"Loss_D: {loss_d.item() if d_updates > 0 else 0:.4f}, Loss_G: {loss_g.item():.4f}, "
                      f"D_acc: {d_acc:.3f}, G_steps: {current_g_steps}")

        # --- Epoch Summary ---
        avg_loss_d = total_loss_d / max(d_updates, 1)
        avg_loss_g = total_loss_g / max(g_updates, 1)
        print(f"=== Epoch [{epoch+1}/{NUM_EPOCHS}] Summary ===")
        print(f"Avg Loss_D: {avg_loss_d:.4f} ({d_updates} updates), Avg Loss_G: {avg_loss_g:.4f} ({g_updates} updates)")
        print(f"Final D_accuracy: {d_acc:.3f}, G_steps_per_D: {current_g_steps}")

        # --- Save generated images ---
        with torch.no_grad():
            generator.eval()
            fake_samples = generator(fixed_noise).detach().cpu()
            save_image(fake_samples, os.path.join(OUTPUT_DIR, 'generated_images', f'epoch_{epoch+1}.png'), 
                      normalize=True, nrow=8)
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