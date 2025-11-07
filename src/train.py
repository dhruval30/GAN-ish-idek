import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision.utils import save_image
import os
from copy import deepcopy 

# Import your models and data loader
from models.generator import Generator
from models.discriminator import Discriminator
from models.forensic_discriminator import ForensicDiscriminator
from utils.data_loader import get_dataloader

# --- EMA Helper Class ---
class EMA:
    """Exponential Moving Average for Generator weights"""
    def __init__(self, model, decay=0.999):
        self.model = deepcopy(model)
        self.model.eval() 
        self.decay = decay

    def update(self, model):
        with torch.no_grad():
            for ema_param, param in zip(self.model.parameters(), model.parameters()):
                ema_param.data.mul_(self.decay).add_(param.data, alpha=1 - self.decay)

# --- Hyperparameters reflecting your strategy ---
LEARNING_RATE_G = 0.001
LEARNING_RATE_D = 0.0002
BETA1 = 0.0
BETA2 = 0.9  # The STABLE Beta2 value
NUM_EPOCHS = 100
LATENT_DIM = 100
BATCH_SIZE = 192
GRADIENT_PENALTY_WEIGHT = 10.0
NUM_D_STEPS = 5
NUM_G_STEPS = 1
EMA_BETA = 0.999 # Standard EMA decay
# --- YOUR IDEA: Start EMA after 5 epochs ---
EMA_START_EPOCH = 10

# --- Dynamic Lambda Schedule ---
LAMBDA_FORENSIC_START = 0.1
LAMBDA_FORENSIC_END = 0.5
LAMBDA_SCHEDULE_EPOCH = 35 # Epoch to increase lambda

# --- Directory setup ---
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(project_root, 'output')
os.makedirs(os.path.join(OUTPUT_DIR, 'generated_images'), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, 'checkpoints'), exist_ok=True)


def gradient_penalty(discriminator, real_images, fake_images, device):
    batch_size = real_images.size(0)
    alpha = torch.rand(batch_size, 1, 1, 1, device=device)
    interpolated = alpha * real_images + (1 - alpha) * fake_images
    interpolated.requires_grad_(True)
    out = discriminator(interpolated)
    gradients = torch.autograd.grad(
        outputs=out, inputs=interpolated,
        grad_outputs=torch.ones_like(out, device=device),
        create_graph=True, retain_graph=True
    )[0]
    gradient_norm = gradients.view(batch_size, -1).norm(2, dim=1)
    penalty = torch.mean((gradient_norm - 1) ** 2)
    return penalty

def compute_discriminator_loss(discriminator, real_images, fake_images, device):
    real_score = discriminator(real_images)
    fake_score = discriminator(fake_images.detach())
    loss_wgan = -torch.mean(real_score) + torch.mean(fake_score)
    gp = gradient_penalty(discriminator, real_images, fake_images.detach(), device)
    return loss_wgan + GRADIENT_PENALTY_WEIGHT * gp

def compute_generator_loss(generator, normal_discriminator, forensic_discriminator, noise, device, current_lambda):
    fake_images = generator(noise)
    normal_score = normal_discriminator(fake_images)
    forensic_score = forensic_discriminator(fake_images)
    return -torch.mean(normal_score) - current_lambda * torch.mean(forensic_score)



def train():
    # --- Device Configuration ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize models
    generator = Generator().to(device)
    normal_discriminator = Discriminator().to(device)
    forensic_discriminator = ForensicDiscriminator().to(device)

    # --- Initialize the EMA generator ---
    print(f"Initializing EMA generator (will start at epoch {EMA_START_EPOCH+1})")
    generator_ema = EMA(generator, decay=EMA_BETA)
    generator_ema.model.to(device)
    ema_active = False # --- Flag to control EMA updates ---

    data_dir = os.path.join(project_root, 'data', 'processed', 'resized')
    
    # --- This assumes your data_loader.py accepts batch_size ---
    dataloader = get_dataloader(data_dir, BATCH_SIZE) 
    print(f"Data loader created with batch size {BATCH_SIZE}.")

    # --- Optimizers with STABLE LRs and Betas ---
    g_optimizer = optim.Adam(generator.parameters(), lr=LEARNING_RATE_G, betas=(BETA1, BETA2))
    d_optimizer = optim.Adam(normal_discriminator.parameters(), lr=LEARNING_RATE_D, betas=(BETA1, BETA2))
    f_optimizer = optim.Adam(forensic_discriminator.parameters(), lr=LEARNING_RATE_D, betas=(BETA1, BETA2))

    # --- Schedulers (still using ReduceLROnPlateau) ---
    g_scheduler = ReduceLROnPlateau(g_optimizer, 'min', factor=0.5, patience=3)
    d_scheduler = ReduceLROnPlateau(d_optimizer, 'min', factor=0.5, patience=3)
    f_scheduler = ReduceLROnPlateau(f_optimizer, 'min', factor=0.5, patience=3)
    
    fixed_noise = torch.randn(64, LATENT_DIM, 1, 1, device=device)

    # ===================================================================
    # --- MAIN ADVERSARIAL TRAINING ---
    # ===================================================================
    print(f"\n--- Starting Main Adversarial Training for {NUM_EPOCHS} epochs ---")
    print(f"G_LR: {LEARNING_RATE_G}, D_LR: {LEARNING_RATE_D}, Betas: ({BETA1}, {BETA2})")
    print(f"Dynamic Lambda: {LAMBDA_FORENSIC_START} (epochs 1-{LAMBDA_SCHEDULE_EPOCH-1}), then {LAMBDA_FORENSIC_END}")
    
    for epoch in range(NUM_EPOCHS):
        total_loss_d_normal, total_loss_d_forensic, total_loss_g = 0.0, 0.0, 0.0
        
        # --- YOUR IDEA: Logic to start EMA ---
        # Note: epoch is 0-indexed, so `epoch == 5` is the *start* of the 6th epoch.
        if not ema_active and epoch == EMA_START_EPOCH:
            print(f"\n--- EMA SYNCING AT EPOCH {epoch+1} ---\n")
            generator_ema.model.load_state_dict(generator.state_dict())
            ema_active = True
        
        # --- Set dynamic lambda for the current epoch ---
        if (epoch + 1) < LAMBDA_SCHEDULE_EPOCH:
            current_lambda = LAMBDA_FORENSIC_START
        else:
            current_lambda = LAMBDA_FORENSIC_END
        
        for i, (real_images, _) in enumerate(dataloader):
            real_images = real_images.to(device)
            current_batch_size = real_images.size(0) 
            
            # --- Train Normal Discriminator ---
            for _ in range(NUM_D_STEPS):
                d_optimizer.zero_grad()
                noise = torch.randn(current_batch_size, LATENT_DIM, 1, 1, device=device)
                fake_images = generator(noise).detach()
                loss_d_normal = compute_discriminator_loss(normal_discriminator, real_images, fake_images, device)
                loss_d_normal.backward()
                d_optimizer.step()
                total_loss_d_normal += loss_d_normal.item()

            # --- Train Forensic Discriminator ---
            for _ in range(NUM_D_STEPS):
                f_optimizer.zero_grad()
                noise = torch.randn(current_batch_size, LATENT_DIM, 1, 1, device=device)
                fake_images = generator(noise).detach()
                loss_d_forensic = compute_discriminator_loss(forensic_discriminator, real_images, fake_images, device)
                loss_d_forensic.backward()
                f_optimizer.step()
                total_loss_d_forensic += loss_d_forensic.item()
            
            # --- Train Generator ---
            for _ in range(NUM_G_STEPS):
                g_optimizer.zero_grad()
                noise = torch.randn(current_batch_size, LATENT_DIM, 1, 1, device=device)
                loss_g = compute_generator_loss(generator, normal_discriminator, forensic_discriminator, noise, device, current_lambda)
                loss_g.backward()
                g_optimizer.step()
                
                # --- Only update EMA if it's active ---
                if ema_active:
                    generator_ema.update(generator)
                
                total_loss_g += loss_g.item()

            if i % 10 == 0 or i == len(dataloader) - 1:
                print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Batch [{i+1}/{len(dataloader)}] "
                      f"Loss_D_Normal: {loss_d_normal.item():.4f}, "
                      f"Loss_D_Forensic: {loss_d_forensic.item():.4f}, "
                      f"Loss_G: {loss_g.item():.4f}")

        # --- Epoch Summary ---
        avg_loss_d_normal = total_loss_d_normal / (len(dataloader) * NUM_D_STEPS)
        avg_loss_d_forensic = total_loss_d_forensic / (len(dataloader) * NUM_D_STEPS)
        avg_loss_g = total_loss_g / (len(dataloader) * NUM_G_STEPS)
        print(f"=== Epoch [{epoch+1}/{NUM_EPOCHS}] Summary ===")
        print(f"Avg Loss_D_Normal: {avg_loss_d_normal:.4f}, "
              f"Avg Loss_D_Forensic: {avg_loss_d_forensic:.4f}, "
              f"Avg Loss_G: {avg_loss_g:.4f}, "
              f"Current Lambda: {current_lambda:.2f}")

        # --- DYNAMIC_LR: Step the schedulers ---
        g_scheduler.step(avg_loss_g)
        d_scheduler.step(avg_loss_d_normal)
        f_scheduler.step(avg_loss_d_forensic)
        
        # --- Save generated images from fixed noise ---
        with torch.no_grad():
            # --- UPDATED: Save from EMA if active, else save from raw generator ---
            if ema_active:
                generator_ema.model.eval()
                fake_samples = generator_ema.model(fixed_noise).detach().cpu()
            else:
                generator.eval()
                fake_samples = generator(fixed_noise).detach().cpu()
                generator.train() # Set back to train mode

            save_image(fake_samples, os.path.join(OUTPUT_DIR, 'generated_images', f'epoch_{epoch+1}.png'), normalize=True, nrow=8)

        # --- Save checkpoints ---
        if (epoch + 1) % 5 == 0:
            checkpoint_dict = {
                'generator_state_dict': generator.state_dict(),
                'normal_discriminator_state_dict': normal_discriminator.state_dict(),
                'forensic_discriminator_state_dict': forensic_discriminator.state_dict(),
                'g_optimizer_state_dict': g_optimizer.state_dict(),
                'd_optimizer_state_dict': d_optimizer.state_dict(),
                'f_optimizer_state_dict': f_optimizer.state_dict(),
                'epoch': epoch + 1,
            }
            
            # --- Only save EMA weights if it's active ---
            if ema_active:
                checkpoint_dict['generator_ema_state_dict'] = generator_ema.model.state_dict()
                
            torch.save(checkpoint_dict, os.path.join(OUTPUT_DIR, 'checkpoints', f'gan_checkpoint_epoch_{epoch+1}.pth'))
            print(f"Checkpoint saved for epoch {epoch+1}")

    print("\nTraining completed successfully!")


if __name__ == '__main__':
    train()