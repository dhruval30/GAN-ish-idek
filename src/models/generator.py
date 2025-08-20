import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm

# Define the network hyperparameters
LATENT_DIM = 100
IMAGE_SIZE = 128 # Changed to 128 to match your dataset
IMAGE_CHANNELS = 3
NGF = 64

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        
        self.main = nn.Sequential(
            # Input is latent vector (100x1x1)
            spectral_norm(nn.ConvTranspose2d(LATENT_DIM, NGF * 16, 4, 1, 0, bias=False)),
            nn.BatchNorm2d(NGF * 16),
            nn.ReLU(True),
            # Size: (NGF*16) x 4 x 4
            
            spectral_norm(nn.ConvTranspose2d(NGF * 16, NGF * 8, 4, 2, 1, bias=False)),
            nn.BatchNorm2d(NGF * 8),
            nn.ReLU(True),
            # Size: (NGF*8) x 8 x 8
            
            spectral_norm(nn.ConvTranspose2d(NGF * 8, NGF * 4, 4, 2, 1, bias=False)),
            nn.BatchNorm2d(NGF * 4),
            nn.ReLU(True),
            # Size: (NGF*4) x 16 x 16
            
            spectral_norm(nn.ConvTranspose2d(NGF * 4, NGF * 2, 4, 2, 1, bias=False)),
            nn.BatchNorm2d(NGF * 2),
            nn.ReLU(True),
            # Size: (NGF*2) x 32 x 32
            
            # --- The new layer to upsample to 128x128 ---
            spectral_norm(nn.ConvTranspose2d(NGF * 2, NGF, 4, 2, 1, bias=False)),
            nn.BatchNorm2d(NGF),
            nn.ReLU(True),
            # Size: NGF x 64 x 64
            
            spectral_norm(nn.ConvTranspose2d(NGF, IMAGE_CHANNELS, 4, 2, 1, bias=False)),
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