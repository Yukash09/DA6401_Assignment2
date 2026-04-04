"""Unified multi-task model
"""

import torch
import torch.nn as nn
from models.classification import VGG11Classifier as VGGC
from models.localization import VGG11Localizer as VGGL
from models.segmentation import VGG11UNet as VGGU

class MultiTaskPerceptionModel(nn.Module):
    """Shared-backbone multi-task model."""

    def __init__(self, num_breeds: int = 37, seg_classes: int = 3, in_channels: int = 3, classifier_path: str = "classifier.pth", localizer_path: str = "localizer.pth", unet_path: str = "unet.pth"):
        """
        Initialize the shared backbone/heads using these trained weights.
        Args:
            num_breeds: Number of output classes for classification head.
            seg_classes: Number of output classes for segmentation head.
            in_channels: Number of input channels.
            classifier_path: Path to trained classifier weights.
            localizer_path: Path to trained localizer weights.
            unet_path: Path to trained unet weights.
        """
        
        super().__init__()

        self.classifier = VGGC(num_classes=num_breeds , in_channels=in_channels)
        self.localizer = VGGL(in_channels=in_channels)
        self.segmentation = VGGU(num_classes=seg_classes , in_channels=in_channels)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.classifier.load_state_dict(torch.load(classifier_path , map_location=device , weights_only=False)['state_dict'])
        self.localizer.load_state_dict(torch.load(localizer_path , map_location=device , weights_only=False)['state_dict'])
        self.segmentation.load_state_dict(torch.load(unet_path , map_location=device , weights_only=False)['state_dict'])

        self.encoder = self.segmentation.encoder

    def forward(self, x: torch.Tensor):
        """Forward pass for multi-task model.
        Args:
            x: Input tensor of shape [B, in_channels, H, W].
        Returns:
            A dict with keys:
            - 'classification': [B, num_breeds] logits tensor.
            - 'localization': [B, 4] bounding box tensor.
            - 'segmentation': [B, seg_classes, H, W] segmentation logits tensor
        """
        # TODO: Implement forward pass.
        # raise NotImplementedError("Implement MultiTaskPerceptionModel.forward")

        enc_output , skip_maps = self.encoder(x , return_features=True)
        classifier_output = self.classifier.fc_block(enc_output)
        localizer_output = self.localizer.regression_decoder(enc_output) * 224.0
        segmentation_output = self.segmentation.split_forward(skip_maps=skip_maps , enc_output=enc_output)

        output_dict = {
            'classification': classifier_output,
            'localization': localizer_output,
            'segmentation': segmentation_output
        }
        
        return output_dict
