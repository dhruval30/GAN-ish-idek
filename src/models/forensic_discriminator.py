# file: models/forensic_discriminator.py

import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm

# Hyperparameters
IMAGE_CHANNELS = 3
NDF = 64

class ForensicDiscriminator(nn.Module):
    def __init__(self):
        super(ForensicDiscriminator, self).__init__()
        
        # --- FIXED_FILTER: Internal non-trainable high-pass filter layer ---
        kernel1 = torch.tensor([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]], dtype=torch.float32)
        kernel2 = torch.tensor([[1,2,1],[0,0,0],[-1,-2,-1]], dtype=torch.float32)
        kernel3 = torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]], dtype=torch.float32)
        self.srm_kernels = torch.stack([kernel1, kernel2, kernel3]).unsqueeze(1)
        self.fixed_filter_layer = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=3, padding=1, bias=False)
        self.fixed_filter_layer.weight = nn.Parameter(self.srm_kernels, requires_grad=False)
        
        # --- Trainable Backbone ---
        self.trainable_backbone = nn.Sequential(
            # Input: 9 channels (3 original channels x 3 filters)
            spectral_norm(nn.Conv2d(IMAGE_CHANNELS * 3, NDF, 4, 2, 1, bias=False)),
            nn.InstanceNorm2d(NDF, affine=True), # --- EXPERIMENTAL: Added InstanceNorm2d ---
            nn.LeakyReLU(0.2, inplace=True),
            
            spectral_norm(nn.Conv2d(NDF, NDF * 2, 4, 2, 1, bias=False)),
            nn.InstanceNorm2d(NDF * 2, affine=True), # --- EXPERIMENTAL: Added InstanceNorm2d ---
            nn.LeakyReLU(0.2, inplace=True),

            spectral_norm(nn.Conv2d(NDF * 2, NDF * 4, 4, 2, 1, bias=False)),
            nn.InstanceNorm2d(NDF * 4, affine=True), # --- EXPERIMENTAL: Added InstanceNorm2d ---
            nn.LeakyReLU(0.2, inplace=True),

            spectral_norm(nn.Conv2d(NDF * 4, NDF * 8, 4, 2, 1, bias=False)),
            nn.InstanceNorm2d(NDF * 8, affine=True), # --- EXPERIMENTAL: Added InstanceNorm2d ---
            nn.LeakyReLU(0.2, inplace=True),

            spectral_norm(nn.Conv2d(NDF * 8, 1, 4, 2, 1, bias=False)),
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.trainable_backbone.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, x):
        # Move the filter layer to the same device as the input tensor
        self.fixed_filter_layer.to(x.device)
        
        r, g, b = torch.unbind(x, dim=1)
        r_res = self.fixed_filter_layer(r.unsqueeze(1))
        g_res = self.fixed_filter_layer(g.unsqueeze(1))
        b_res = self.fixed_filter_layer(b.unsqueeze(1))
        residuals = torch.cat([r_res, g_res, b_res], dim=1)
        out = self.trainable_backbone(residuals)
        return out.view(-1)