"""Localization modules
"""

import torch
import torch.nn as nn
from models.vgg11 import VGG11Encoder as VGGE
from models.layers import CustomDropout as CD

class VGG11Localizer(nn.Module):
    """VGG11-based localizer."""

    def __init__(self, in_channels: int = 3, dropout_p: float = 0.5):
        """
        Initialize the VGG11Localizer model.

        Args:
            in_channels: Number of input channels.
            dropout_p: Dropout probability for the localization head.
        """
        
        super().__init__()
        self.encoder = VGGE(in_channels=in_channels , batch_norm=True)

        self.regression_decoder = nn.Sequential(
            nn.Flatten() , 
            nn.Linear(512 * 7 * 7 , 4096) , 
            nn.ReLU(inplace=True) , 
            CD(p=dropout_p) , 

            nn.Linear(4096 , 1024) , 
            nn.ReLU(inplace=True) , 
            CD(p=0.5) ,

            nn.Linear(1024 , 4) , 
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for localization model.
        Args:
            x: Input tensor of shape [B, in_channels, H, W].

        Returns:
            Bounding box coordinates [B, 4] in (x_center, y_center, width, height) format in original image pixel space(not normalized values).
        """
        # TODO: Implement forward pass.
        # raise NotImplementedError("Implement VGG11Localizer.forward")

        enc_output = self.encoder(x) 
        output = self.regression_decoder(enc_output)
        scaled_output = output * 224.0 

        return scaled_output
    
    def load_pth(self , checkpoint:str , device:torch.device):
        checkpoint_dict = torch.load(checkpoint , map_location=device)
        weights = checkpoint_dict.get("state_dict") 

        enc_weights = {}
        for key , value in weights.items():
            if key.startswith("encoder."):
                new_key = key.replace("encoder." , "")
                enc_weights[new_key] = value 
        
        self.encoder.load_state_dict(enc_weights)
