import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm

# Define the network hyperparameters
IMAGE_SIZE = 128
IMAGE_CHANNELS = 3
NDF = 64

class ResBlockDown(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResBlockDown, self).__init__()
        self.conv1 = spectral_norm(nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False))
        self.conv2 = spectral_norm(nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False))
        self.downsample = nn.AvgPool2d(2)
        
        self.shortcut = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels, out_channels, 1, 1, bias=False))
        )

    def forward(self, x):
        out = self.conv1(x)
        out = nn.LeakyReLU(0.2)(out)
        out = self.conv2(out)
        out = self.downsample(out)
        out += self.downsample(self.shortcut(x))
        return nn.LeakyReLU(0.2)(out)

class ForensicDiscriminator(nn.Module):
    def __init__(self):
        super(ForensicDiscriminator, self).__init__()
        
        self.main = nn.Sequential(
            # Input is IMAGE_CHANNELS x 128 x 128
            ResBlockDown(IMAGE_CHANNELS, NDF),
            # Size: NDF x 64 x 64
            ResBlockDown(NDF, NDF * 2),
            # Size: (NDF*2) x 32 x 32
            ResBlockDown(NDF * 2, NDF * 4),
            # Size: (NDF*4) x 16 x 16
            ResBlockDown(NDF * 4, NDF * 8),
            # Size: (NDF*8) x 8 x 8
            
            spectral_norm(nn.Conv2d(NDF * 8, 1, 4, 1, 0, bias=False))
            # No Sigmoid activation here!
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, input):
        features = self.main(input)
        # Take the mean over the spatial dimensions to get a single probability per image
        output = torch.mean(features, dim=(2, 3))
        
        # Flatten the output to a 1D tensor
        return output.view(-1)