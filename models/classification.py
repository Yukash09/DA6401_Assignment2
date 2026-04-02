"""Classification components
"""

import torch
import torch.nn as nn
from vgg11 import VGG11Encoder as VGE
from layers import CustomDropout as CD


class VGG11Classifier(nn.Module):
    """Full classifier = VGG11Encoder + ClassificationHead."""

    def __init__(self, num_classes: int = 37, in_channels: int = 3, dropout_p: float = 0.5 , batch_norm: bool = False):
        """
        Initialize the VGG11Classifier model.
        Args:
            num_classes: Number of output classes.
            in_channels: Number of input channels.
            dropout_p: Dropout probability for the classifier head.
        """
        
        self.encoder = VGE(in_channels=in_channels , batch_norm=batch_norm)

        self.fc_block = nn.Sequential(
            nn.Flatten() , 
            nn.Linear(in_features=512 * 7 * 7 , out_features=4096) , 
            nn.ReLU(inplace=True) , 
            CD(p=dropout_p) , 

            nn.Linear(4096 , 4096) , 
            nn.ReLU(inplace=True) , 
            CD(p=dropout_p) ,

            nn.Linear(4096, num_classes) , 
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for classification model.
        Args:
            x: Input tensor of shape [B, in_channels, H, W].
        Returns:
            Classification logits [B, num_classes].
        """
        # TODO: Implement forward pass.
        # raise NotImplementedError("Implement VGG11Classifier.forward")

        conv_output = self.encoder(x) 
        class_logits = self.fc_block(conv_output) 

        return class_logits
