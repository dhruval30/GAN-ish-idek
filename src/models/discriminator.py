# file: models/discriminator.py

import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm

# Hyperparameters
IMAGE_CHANNELS = 3
NDF = 64

class ResBlockDown(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResBlockDown, self).__init__()
        self.conv1 = spectral_norm(nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False))
        self.conv2 = spectral_norm(nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False))
        self.downsample = nn.AvgPool2d(2)
        
        # --- EXPERIMENTAL: Added InstanceNorm2d for potential stability ---
        self.in1 = nn.InstanceNorm2d(out_channels, affine=True)
        self.in2 = nn.InstanceNorm2d(out_channels, affine=True)
        
        self.shortcut = nn.Sequential(
            nn.AvgPool2d(2),
            spectral_norm(nn.Conv2d(in_channels, out_channels, 1, 1, bias=False))
        )

    def forward(self, x):
        shortcut_out = self.shortcut(x)
        
        out = self.conv1(x)
        out = self.in1(out) # Apply InstanceNorm
        out = nn.LeakyReLU(0.2, inplace=True)(out)
        
        out = self.conv2(out)
        out = self.in2(out) # Apply InstanceNorm
        out = self.downsample(out)
        
        out += shortcut_out
        return nn.LeakyReLU(0.2, inplace=True)(out)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        
        self.main = nn.Sequential(
            # --- ARCH_MISMATCH_FIX: Correctly handles 128x128 images ---
            # Input: 3 x 128 x 128
            ResBlockDown(IMAGE_CHANNELS, NDF),      # -> NDF x 64 x 64
            ResBlockDown(NDF, NDF * 2),             # -> NDF*2 x 32 x 32
            ResBlockDown(NDF * 2, NDF * 4),         # -> NDF*4 x 16 x 16
            ResBlockDown(NDF * 4, NDF * 8),         # -> NDF*8 x 8 x 8
            ResBlockDown(NDF * 8, NDF * 8),         # -> NDF*8 x 4 x 4
            spectral_norm(nn.Conv2d(NDF * 8, 1, 4, 1, 0, bias=False)) # -> 1 x 1 x 1
        )
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, input):
        return self.main(input).view(-1)