import torch
import os
from torchvision.utils import save_image

from models.generator import Generator

CHECKPOINT_PATH = '/Users/dhruval/Documents/GAN-ish-idek/output/checkpoints/gan_checkpoint_epoch_50.pth'
NUM_IMAGES = 1 


def run_inference():
    """
    Loads a trained generator and generates new images.
    """
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    project_root = os.path.dirname(os.path.abspath(__file__))
    CHECKPOINT_DIR = os.path.join(project_root, 'output', 'checkpoints')
    INFERENCE_DIR = os.path.join(project_root, 'output-inference-img', 'inference')
    os.makedirs(INFERENCE_DIR, exist_ok=True)
    
    # --- Model and Checkpoint Loading ---
    generator = Generator().to(device)
    full_checkpoint_path = os.path.join(CHECKPOINT_DIR, CHECKPOINT_PATH)
    
    print(f"Loading checkpoint from {full_checkpoint_path}")
    checkpoint = torch.load(full_checkpoint_path, map_location=device)
    generator.load_state_dict(checkpoint['generator_state_dict'])
    generator.eval()
    
    print(f"Generating {NUM_IMAGES} new image(s)...")
    
    # --- Generate images ---
    with torch.no_grad():
        noise = torch.randn(NUM_IMAGES, 100, 1, 1, device=device)
        fake_images = generator(noise).detach().cpu()
        
    # --- Save the images ---
    if NUM_IMAGES == 1:
        output_filename = os.path.join(INFERENCE_DIR, f'generated_image_{checkpoint["epoch"]}.png')
        save_image(fake_images.squeeze(0), output_filename, normalize=True)
    else:
        output_filename = os.path.join(INFERENCE_DIR, f'generated_images_{checkpoint["epoch"]}.png')
        save_image(fake_images, output_filename, normalize=True, nrow=8)

    print(f"Generated image(s) saved to {output_filename}")


if __name__ == '__main__':
    run_inference()