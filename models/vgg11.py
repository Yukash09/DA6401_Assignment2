"""VGG11 encoder
"""

from typing import Dict, Tuple, Union

import torch
import torch.nn as nn


class VGG11Encoder(nn.Module):
    """VGG11-style encoder with optional intermediate feature returns.
    """

    def __init__(self, in_channels: int = 3 , batch_norm: bool = False):
        """Initialize the VGG11Encoder model."""
        
        super().__init__()

        self.conv_block1 = self.conv_block(in_channels=in_channels , out_channels=64 , kernel_size=3 , padding=1 , batch_norm=batch_norm)

        self.maxpool_1 = nn.MaxPool2d(kernel_size=2 , stride=2)

        self.conv_block2 = self.conv_block(in_channels=64 , out_channels=128, kernel_size=3 , padding=1 , batch_norm=batch_norm)

        self.maxpool_2 = nn.MaxPool2d(kernel_size=2 , stride=2)

        self.conv_block3 = self.conv_block(in_channels=128 , out_channels=256 , kernel_size=3 , padding=1 , batch_norm=batch_norm)
        self.conv_block4 = self.conv_block(in_channels=256 , out_channels=256, kernel_size=3 , padding=1 , batch_norm=batch_norm)

        self.maxpool_3 = nn.MaxPool2d(kernel_size=2 , stride=2)

        self.conv_block5 = self.conv_block(in_channels=256 , out_channels=512 , kernel_size=3 , padding=1 , batch_norm=batch_norm)
        self.conv_block6 = self.conv_block(in_channels=512 , out_channels=512, kernel_size=3 , padding=1 , batch_norm=batch_norm)

        self.maxpool_4 = nn.MaxPool2d(kernel_size=2 , stride=2)

        self.conv_block7 = self.conv_block(in_channels=512 , out_channels=512 , kernel_size=3 , padding=1 , batch_norm=batch_norm)
        self.conv_block8 = self.conv_block(in_channels=512 , out_channels=512, kernel_size=3 , padding=1 , batch_norm=batch_norm)

        self.maxpool_5 = nn.MaxPool2d(kernel_size=2 , stride=2)     

    def conv_block(self , in_channels: int , out_channels: int , kernel_size: int , padding: int , batch_norm: bool = True):
        if batch_norm:
            block = nn.Sequential(
                nn.Conv2d(in_channels=in_channels , out_channels=out_channels , kernel_size=kernel_size , padding=padding) , 
                nn.BatchNorm2d(out_channels) , 
                nn.ReLU(inplace=True)
                )
        else:
            block = nn.Sequential(
                nn.Conv2d(in_channels=in_channels , out_channels=out_channels , kernel_size=kernel_size , padding=padding) , 
                # nn.BatchNorm2d(out_channels) , 
                nn.ReLU(inplace=True)
                )
        
        return block

    def forward(
        self, x: torch.Tensor, return_features: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """Forward pass.

        Args:
            x: input image tensor [B, 3, H, W].
            return_features: if True, also return skip maps for U-Net decoder.

        Returns:
            - if return_features=False: bottleneck feature tensor.
            - if return_features=True: (bottleneck, feature_dict).
        """
        # TODO: Implement forward pass.
        # raise NotImplementedError("Implement VGG11Encoder.forward")
        
        layer1 = self.conv_block1(x)
        layer2 = self.maxpool_1(layer1)

        layer3 = self.conv_block2(layer2)
        layer4 = self.maxpool_2(layer3)

        layer5 = self.conv_block3(layer4)
        layer6 = self.conv_block4(layer5)
        layer7 = self.maxpool_3(layer6)

        layer8 = self.conv_block5(layer7)
        layer9 = self.conv_block6(layer8)
        layer10 = self.maxpool_4(layer9)

        layer11 = self.conv_block7(layer10)
        layer12 = self.conv_block8(layer11)
        layer13 = self.maxpool_5(layer12)

        if return_features:
            # TODO: Need to return proper dict, idk what to do here.
            return (layer13 , {})
        
        return layer13
    




