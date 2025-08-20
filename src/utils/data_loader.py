import os
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader


IMAGE_SIZE = 128

BATCH_SIZE = 64

transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

def get_dataloader(data_dir):
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4
    )
    return dataloader

if __name__ == '__main__':
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(project_root, 'data', 'processed', 'resized')
    dataloader = get_dataloader(data_dir)
    print("DataLoader test successful.")