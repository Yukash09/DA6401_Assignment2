"""Segmentation model
"""

import torch
import torch.nn as nn
from models.vgg11 import VGG11Encoder as VGGE

class VGG11UNet(nn.Module):
    """U-Net style segmentation network.
    """

    def __init__(self, num_classes: int = 3, in_channels: int = 3, dropout_p: float = 0.5):
        """
        Initialize the VGG11UNet model.

        Args:
            num_classes: Number of output classes.
            in_channels: Number of input channels.
            dropout_p: Dropout probability for the segmentation head.
        """
        
        super().__init__()

        self.encoder = VGGE(in_channels=in_channels , batch_norm=True)

        # out_channels from conv_block8 --> in_channels for transpose conv
        # Similary in_channel --> out_channel , upsampling 
        self.up_conv_block1 = nn.ConvTranspose2d(in_channels=512 , out_channels=512 , kernel_size=2 , stride=2)
    
        # This part should have skip connections , so we need to account for that , 512 + 512 = 1024 (conv_block 8 --> 512 to 512)
        self.conv_block1_1 = self.conv_block(in_channels=1024 , out_channels=512 , kernel_size=3 , padding=1)
        self.conv_block1_2 = self.conv_block(in_channels=512 , out_channels=512 , kernel_size=3 , padding=1)

        # after this conv_block , the grid is now 7x7 --> 14x14 

        self.up_conv_block2 = nn.ConvTranspose2d(in_channels=512 , out_channels=256 , kernel_size=2 , stride=2)

        # We get 256 out_channels from transpose conv , and conv_block6 is the symmetrical block with 512 out_channels so 256+512 = 768 
        self.conv_block_2_1 = self.conv_block(in_channels=768 , out_channels=256 , kernel_size=3 , padding=1)
        self.conv_block_2_2 = self.conv_block(in_channels=256 , out_channels=256 , kernel_size=3 , padding=1)
        # after this conv_block , the grid is now 14x14 --> 28x28 

        self.up_conv_block3 = nn.ConvTranspose2d(in_channels=256 , out_channels=128 , kernel_size=2 , stride=2)

        # Now its 128 + 256 = 384 
        self.conv_block_3_1 = self.conv_block(in_channels=384 , out_channels=128 , kernel_size=3 , padding=1)
        self.conv_block_3_2 = self.conv_block(in_channels=128 , out_channels=128 , kernel_size=3 , padding=1)

        self.up_conv_block4 = nn.ConvTranspose2d(in_channels=128 , out_channels=64 , kernel_size=2 , stride=2)

        # Here its 64 + 128 = 192
        self.conv_block_4_1 = self.conv_block(in_channels=192 , out_channels=64 , kernel_size=3 , padding=1)
        self.conv_block_4_2 = self.conv_block(in_channels=64 , out_channels=64 , kernel_size=3 , padding=1)

        self.up_conv_block5 = nn.ConvTranspose2d(in_channels=64 , out_channels=64 , kernel_size=2 , stride=2)

        # Here its 64 + 64 = 128 
        self.conv_block_5_1 = self.conv_block(in_channels=128 ,out_channels=64 , kernel_size=3 , padding=1)
        self.conv_block_5_2 = self.conv_block(in_channels=64 , out_channels=64 , kernel_size=3 , padding=1)
        # Now its (2^5 * 2^5)* (7 x 7) = (224 x 224) , Information cannot be created or destroyed, It can only be transferred - Sun Tzu.

        self.up_conv_block6 = nn.ConvTranspose2d(in_channels=64 , out_channels=num_classes , kernel_size=1 , stride=1)        

    def conv_block(self , in_channels:int , out_channels:int , kernel_size:int , padding:int , batch_norm: bool = True):
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
    
    def split_forward(self , skip_maps , enc_output):
        rev_layer1 = self.up_conv_block1(enc_output)
        rev_layer2_1 = self.conv_block1_1(torch.cat([rev_layer1 , skip_maps["skip_map5"]] , dim=1))
        rev_layer2_2 = self.conv_block1_2(rev_layer2_1) 

        rev_layer3 = self.up_conv_block2(rev_layer2_2)
        rev_layer4_1 = self.conv_block_2_1(torch.cat([rev_layer3 , skip_maps["skip_map4"]] , dim=1))
        rev_layer4_2 = self.conv_block_2_2(rev_layer4_1) 

        rev_layer5 = self.up_conv_block3(rev_layer4_2)
        rev_layer6_1 = self.conv_block_3_1(torch.cat([rev_layer5 , skip_maps["skip_map3"]] , dim=1))
        rev_layer6_2 = self.conv_block_3_2(rev_layer6_1) 

        rev_layer7 = self.up_conv_block4(rev_layer6_2)
        rev_layer8_1 = self.conv_block_4_1(torch.cat([rev_layer7 , skip_maps["skip_map2"]] , dim=1))
        rev_layer8_2 = self.conv_block_4_2(rev_layer8_1) 

        rev_layer9 = self.up_conv_block5(rev_layer8_2)
        rev_layer10_1 = self.conv_block_5_1(torch.cat([rev_layer9 , skip_maps["skip_map1"]] , dim=1))
        rev_layer10_2 = self.conv_block_5_2(rev_layer10_1) 

        logits = self.up_conv_block6(rev_layer10_2)

        return logits


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for segmentation model.
        Args:
            x: Input tensor of shape [B, in_channels, H, W].

        Returns:
            Segmentation logits [B, num_classes, H, W].
        """
        # TODO: Implement forward pass.
        # raise NotImplementedError("Implement VGG11UNet.forward")

        enc_output , skip_maps = self.encoder(x , return_features=True)

        logits = self.split_forward(skip_maps=skip_maps , enc_output=enc_output)

        return logits

    def load_pth(self , checkpoint:str , device:torch.device):
        checkpoint_dict = torch.load(checkpoint , map_location=device , weights_only=False)
        weights = checkpoint_dict.get("state_dict") 

        enc_weights = {}
        for key , value in weights.items():
            if key.startswith("encoder."):
                new_key = key.replace("encoder." , "")
                enc_weights[new_key] = value 
        
        self.encoder.load_state_dict(enc_weights)
