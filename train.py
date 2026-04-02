"""Training entrypoint
"""
import wandb
import albumentations as A
from albumentations.pytorch import ToTensorV2 
from data.pets_dataset import OxfordIIITPetDataset
from models.classification import VGG11Classifier as VGGC

transform = A.Compose([
    A.Resize(224 , 224),
    A.HorizontalFlip(),
    A.ShiftScaleRotate(
        shift_limit=0.05,
        scale_limit=0.1,
        rotate_limit=15,
        p=0.5
    ),
    A.RandomBrightnessContrast(),
    A.Normalize(),
    ToTensorV2()
] , bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))

# training_data = OxfordIIITPetDataset(True , transform)

# print("Total training images: " , len(training_data))
# img, id, bbox, segment = training_data[0]
# print("Image shape: " , img.shape) 
# print("Bounding Box: " , bbox)

def classifier(batch_norm:bool , dropout:int):
  wandb.init(
    project="DA6401_Assignment2" , 
    name="Classifier" , 
    config = {
      "dropout": dropout , 
      "batch_norm": batch_norm
    }
  )

  model = VGGC(
    num_classes=37, 
    in_channels=3,
    dropout_p=dropout, 
    batch_norm=batch_norm
  )