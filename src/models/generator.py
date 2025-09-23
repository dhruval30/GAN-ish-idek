import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm

# Define the network hyperparameters
LATENT_DIM = 100
IMAGE_SIZE = 128
IMAGE_CHANNELS = 3
NGF = 64

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResBlock, self).__init__()
        self.conv1 = spectral_norm(nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False))
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(True)
        self.conv2 = spectral_norm(nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False))
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Skip connection
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                spectral_norm(nn.Conv2d(in_channels, out_channels, 1, 1, bias=False)),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return self.relu(out)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        
        self.main = nn.Sequential(
            # Input is latent vector (100x1x1)
            spectral_norm(nn.ConvTranspose2d(LATENT_DIM, NGF * 16, 4, 1, 0, bias=False)),
            nn.BatchNorm2d(NGF * 16),
            nn.ReLU(True),
            # Size: (NGF*16) x 4 x 4
            
            # Upsample to 8x8
            nn.Upsample(scale_factor=2, mode='nearest'),
            ResBlock(NGF * 16, NGF * 8),
            # Size: (NGF*8) x 8 x 8
            
            # Upsample to 16x16
            nn.Upsample(scale_factor=2, mode='nearest'),
            ResBlock(NGF * 8, NGF * 4),
            # Size: (NGF*4) x 16 x 16
            
            # Upsample to 32x32
            nn.Upsample(scale_factor=2, mode='nearest'),
            ResBlock(NGF * 4, NGF * 2),
            # Size: (NGF*2) x 32 x 32
            
            # Upsample to 64x64
            nn.Upsample(scale_factor=2, mode='nearest'),
            ResBlock(NGF * 2, NGF),
            # Size: NGF x 64 x 64
            
            # Upsample to 128x128
            nn.Upsample(scale_factor=2, mode='nearest'),
            ResBlock(NGF, NGF),
            # Size: NGF x 128 x 128

            spectral_norm(nn.Conv2d(NGF, IMAGE_CHANNELS, 3, 1, 1, bias=False)),
            nn.Tanh()
            # Final output size: IMAGE_CHANNELS x 128 x 128
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights using best practices"""
        for m in self.modules():
            if isinstance(m, (nn.ConvTranspose2d, nn.Conv2d)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, input):
        return self.main(input)