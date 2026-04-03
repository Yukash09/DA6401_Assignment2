"""Custom IoU loss 
"""

import torch
import torch.nn as nn

class IoULoss(nn.Module):
    """IoU loss for bounding box regression.
    """

    def __init__(self, eps: float = 1e-6, reduction: str = "mean"):
        """
        Initialize the IoULoss module.
        Args:
            eps: Small value to avoid division by zero.
            reduction: Specifies the reduction to apply to the output: 'mean' | 'sum'.
        """
        super().__init__()
        self.eps = eps
        self.reduction = reduction

        if not (reduction == "mean" or reduction == "sum"):
            raise ValueError("Invalid reduction type")
        # TODO: validate reduction in {"none", "mean", "sum"}.

    def forward(self, pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
        """Compute IoU loss between predicted and target bounding boxes.
        Args:
            pred_boxes: [B, 4] predicted boxes in (x_center, y_center, width, height) format.
            target_boxes: [B, 4] target boxes in (x_center, y_center, width, height) format."""
        # TODO: implement IoU loss.
        # raise NotImplementedError("Implement IoULoss.forward")

        pred = [
            pred_boxes[: , 0] - (pred_boxes[: , 2]/2) ,  # xmin
            pred_boxes[: , 1] - (pred_boxes[: , 3]/2) ,  # ymin
            pred_boxes[: , 0] + (pred_boxes[: , 2]/2) ,  # xmax
            pred_boxes[: , 1] + (pred_boxes[: , 3]/2)    # ymax
        ]

        target = [
            target_boxes[: , 0] - (target_boxes[: , 2]/2) ,  # xmin
            target_boxes[: , 1] - (target_boxes[: , 3]/2) ,  # ymin
            target_boxes[: , 0] + (target_boxes[: , 2]/2) ,  # xmax
            target_boxes[: , 1] + (target_boxes[: , 3]/2)    # ymax
        ]

        int_box = [
            torch.max(pred[0] , target[0]) , 
            torch.max(pred[1] , target[1]) ,
            torch.min(pred[2] , target[2]) , 
            torch.min(pred[3] , target[3]) 
        ]

        intersection_area = max(0 , (int_box[2] - int_box[0]) * (int_box[3] - int_box[1]))

        pred_area = pred_boxes[: , 2] * pred_boxes[: , 3]
        target_area = target_boxes[: , 2] * target_boxes[: , 3]
        union_area = pred_area + target_area - intersection_area 

        IoU = intersection_area / (union_area + self.eps)

        loss = 1 - IoU 

        if self.reduction == "mean":
            return loss.mean()
        else:
            return loss.sum()