import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm

# Define the network hyperparameters
LATENT_DIM = 100  # Size of the input noise vector
IMAGE_SIZE = 64   # Target image size
IMAGE_CHANNELS = 3  # RGB images
NGF = 64  # Size of feature maps in the generator

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        
        # Use spectral normalization for better training stability
        self.main = nn.Sequential(
            # Input is latent vector (100x1x1)
            spectral_norm(nn.ConvTranspose2d(LATENT_DIM, NGF * 8, 4, 1, 0, bias=False)),
            nn.BatchNorm2d(NGF * 8),
            nn.ReLU(True),
            # Size: (NGF*8) x 4 x 4
            
            spectral_norm(nn.ConvTranspose2d(NGF * 8, NGF * 4, 4, 2, 1, bias=False)),
            nn.BatchNorm2d(NGF * 4),
            nn.ReLU(True),
            # Size: (NGF*4) x 8 x 8
            
            spectral_norm(nn.ConvTranspose2d(NGF * 4, NGF * 2, 4, 2, 1, bias=False)),
            nn.BatchNorm2d(NGF * 2),
            nn.ReLU(True),
            # Size: (NGF*2) x 16 x 16
            
            spectral_norm(nn.ConvTranspose2d(NGF * 2, NGF, 4, 2, 1, bias=False)),
            nn.BatchNorm2d(NGF),
            nn.ReLU(True),
            # Size: NGF x 32 x 32
            
            spectral_norm(nn.ConvTranspose2d(NGF, IMAGE_CHANNELS, 4, 2, 1, bias=False)),
            nn.Tanh()
            # Final output size: IMAGE_CHANNELS x 64 x 64
        )
        
        # Initialize weights using Xavier/Glorot initialization
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