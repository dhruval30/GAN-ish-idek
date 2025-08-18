import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm

# Define the network hyperparameters
IMAGE_SIZE = 64
IMAGE_CHANNELS = 3
NDF = 48  # Reduced from 64 to 48 to decrease discriminator capacity

class Discriminator(nn.Module):
    def __init__(self, use_spectral_norm=True):
        super(Discriminator, self).__init__()
        
        # Helper function to optionally apply spectral normalization
        def maybe_spectral_norm(layer):
            return spectral_norm(layer) if use_spectral_norm else layer
        
        # Shared convolutional layers to extract features (reduced capacity)
        self.shared_layers = nn.Sequential(
            # Input is IMAGE_CHANNELS x 64 x 64
            maybe_spectral_norm(nn.Conv2d(IMAGE_CHANNELS, NDF, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),  # Add dropout for regularization
            # Size: NDF x 32 x 32
            
            maybe_spectral_norm(nn.Conv2d(NDF, NDF * 2, 4, 2, 1, bias=False)),
            nn.BatchNorm2d(NDF * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            # Size: (NDF*2) x 16 x 16
            
            maybe_spectral_norm(nn.Conv2d(NDF * 2, NDF * 4, 4, 2, 1, bias=False)),
            nn.BatchNorm2d(NDF * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            # Size: (NDF*4) x 8 x 8
            
            # Removed one layer to reduce capacity further
            maybe_spectral_norm(nn.Conv2d(NDF * 4, NDF * 4, 4, 2, 1, bias=False)),  # Keep same channels
            nn.BatchNorm2d(NDF * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            # Final shared feature size: (NDF*4) x 4 x 4
        )
        
        # Output head for the "normal" detector
        self.normal_detector = nn.Sequential(
            maybe_spectral_norm(nn.Conv2d(NDF * 4, 1, 4, 1, 0, bias=False)),
            nn.Sigmoid()
        )
        
        # Output head for the "forensic" detector
        self.forensic_detector = nn.Sequential(
            maybe_spectral_norm(nn.Conv2d(NDF * 4, 1, 4, 1, 0, bias=False)),
            nn.Sigmoid()
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights using best practices"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, input):
        # Process the input image through the shared layers
        features = self.shared_layers(input)
        
        # Get outputs from both detectors
        normal_output = self.normal_detector(features)
        forensic_output = self.forensic_detector(features)
        
        # Take the mean over the spatial dimensions to get a single probability per image
        normal_output = torch.mean(normal_output, dim=(2, 3))
        forensic_output = torch.mean(forensic_output, dim=(2, 3))
        
        # Flatten the output to a 1D tensor
        return normal_output.view(-1), forensic_output.view(-1)
    
    def get_features(self, input):
        """Return intermediate features for feature matching loss if needed"""
        return self.shared_layers(input)