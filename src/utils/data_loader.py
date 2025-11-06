# file: utils/data_loader.py

import os
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

IMAGE_SIZE = 128

transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# --- Make sure this accepts batch_size ---
def get_dataloader(data_dir, batch_size):
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size, # Use the provided batch_size
        shuffle=True,
        num_workers=4
    )
    return dataloader

if __name__ == '__main__':
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(project_root, 'data', 'processed', 'resized')
    # Test with a default batch size
    dataloader = get_dataloader(data_dir, batch_size=64) 
    print("DataLoader test successful.")