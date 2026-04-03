"""Training entrypoint
"""
# import wandb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader , random_split
import torch.optim as optim
import albumentations as A
from albumentations.pytorch import ToTensorV2 
from data.pets_dataset import OxfordIIITPetDataset
from models.classification import VGG11Classifier as VGGC
from models.localization import VGG11Localizer as VGGL
from models.segmentation import VGG11UNet as VGGU
from losses.iou_loss import IoULoss

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

test_transform = A.Compose([
  A.Resize(224 , 224),
  A.Normalize(),
  ToTensorV2()
], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))

# training_data = OxfordIIITPetDataset(True , transform)

# print("Total training images: " , len(training_data))
# img, id, bbox, segment = training_data[0]
# print("Image shape: " , img.shape) 
# print("Bounding Box: " , bbox)

def classifier(batch_norm:bool , dropout):
  # wandb.init(
  #   project="DA6401_Assignment2" , 
  #   name="Classifier" , 
  #   config = {
  #     "dropout": dropout , 
  #     "batch_norm": batch_norm
  #   }
  # )

  # config = wandb.config 
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model = VGGC(
    num_classes=37, 
    in_channels=3,
    dropout_p=dropout, 
    batch_norm=batch_norm
  ).to(device)

  training_data = OxfordIIITPetDataset(isTrain=True , transform=transform) 
  train_loader = DataLoader(training_data , batch_size=16 , shuffle=True , num_workers=4)

  test_data = OxfordIIITPetDataset(isTrain=False, transform=test_transform)
  test_loader = DataLoader(test_data , batch_size=16 , shuffle=False , num_workers=4)

  loss_fn = nn.CrossEntropyLoss()
  optimizer = optim.Adam(model.parameters() , lr=0.0001)
  scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max' , factor=0.5 , patience=3)

  best_acc = 0.0 

  for epoch in range(40):

    print(f"Epoch: {epoch}")
    model.train()
    epoch_loss = 0.0 
    total_img = 0 
    correct_class = 0 

    for idx , (images , ids , bboxs , segments) in enumerate(train_loader):
      images , ids = images.to(device) , ids.to(device)
      logits = model(images) 
      loss = loss_fn(logits , ids)

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      epoch_loss += loss.item()

      _ , predicted = torch.max(logits.data , 1)
      total_img += ids.size(0)
      correct_class += (predicted == ids).sum().item()

      if (idx + 1) % 20 == 0:
        print(f"Batch {idx}/{len(train_loader)} , Loss: {loss.item():.4f}")

    train_accuracy = correct_class/total_img * 100 
    train_loss = epoch_loss / len(train_loader)

    print(f"Train accuracy: {train_accuracy} , Train loss: {train_loss}")

    model.eval()
    val_loss = 0.0 
    correct_val = 0 
    total_val = 0 

    with torch.no_grad():
      for images , ids , bboxs , segments in test_loader:
        images , ids = images.to(device) , ids.to(device)
        logits = model(images)
        loss = loss_fn(logits , ids)

        val_loss += loss.item()
        _ , predicted = torch.max(logits.data , 1)
        total_val += ids.size(0)
        correct_val += (predicted == ids).sum().item()

    val_acc = correct_val / total_val * 100 
    val_loss = val_loss / len(test_loader)

    scheduler.step(val_acc)

    print(f"Validation Loss: {val_loss:.4f} , Validation Accuracy: {val_acc:.2f}") 

    if val_acc > best_acc:
      best_acc = val_acc 
      checkpoint = {
        "state_dict": model.state_dict() , 
        "epoch": epoch ,
        "best_metric": best_acc
      }
      torch.save(checkpoint , "./checkpoints/classifier.pth")

def localizer(batch_norm:bool , dropout):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VGGL(
      in_channels=3,
      dropout_p=0.5
    ).to(device)

    model.load_pth("./checkpoints/classifier.pth" , device)

    training_data = OxfordIIITPetDataset(isTrain=True , transform=transform)

    # train_data = training_data[:int(0.8*len(training_data))]
    # val_data = training_data[int(0.8*len(training_data)):]

    generator = torch.Generator().manual_seed(3)
    train_data , val_data = random_split(training_data , [int(0.8*len(training_data)) , len(training_data) - int(0.8*len(training_data))] , generator)

    print(f"Training samples: {len(train_data)} and Validation samples:{len(val_data)}")

    train_loader = DataLoader(train_data , batch_size=16 , shuffle=True)
    val_loader = DataLoader(val_data , batch_size=16 , shuffle=False)

    loss_fn = IoULoss() 
    optimizer = optim.Adam(model.parameters() , lr=0.0001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max' , factor=0.5 , patience=3)

    best_loss = 1e18 

    for epoch in range(40):

      print(f"Epoch: {epoch}") 
      model.train()
      epoch_loss = 0.0 
      
      for idx , (images , ids , bboxs , segments) in enumerate(train_loader):
        # images , bboxs = images.to(device) , bboxs.to(device)
        images = images.to(device)
        bboxs = (bboxs.float()).to(device)
        output = model(images)
        loss = loss_fn(output , bboxs)

        optimizer.zero_grad()
        loss.backward() 
        optimizer.step()

        epoch_loss += loss.item()

        if (idx + 1)%20 == 0:
          print(f"Batch {idx + 1}/{len(train_loader)} , Loss: {loss.item():.4f}")

      train_loss = epoch_loss / len(train_loader)

      print(f"Train Loss: {train_loss}")

      model.eval()
      val_loss = 0.0 

      with torch.no_grad():
        for images , ids , bboxs , segments in val_loader:
          # images , bboxs = images.to(device) , bboxs.to(device)
          images = images.to(device)
          bboxs = (bboxs.float()).to(device)
          output = model(images)
          loss = loss_fn(output , bboxs)
          val_loss += loss.item()


      val_loss = val_loss / len(val_loader)
      scheduler.step(val_loss)

      if val_loss < best_loss:
        best_loss = val_loss 
        checkpoint = {
          "state_dict": model.state_dict() , 
          "epoch": epoch ,
          "best_metric": best_loss        
        }
        torch.save(checkpoint , "./checkpoints/localizer.pth")

      print(f"Validation Loss: {val_loss:.4f}")


def segmentation(batch_norm:bool , dropout):
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model = VGGU(
    in_channels=3,
    dropout_p=0.5
  ).to(device)

  model.load_pth("./checkpoints/classifier.pth" , device)

  training_data = OxfordIIITPetDataset(isTrain=True , transform=transform)

  generator = torch.Generator().manual_seed(3)
  train_data , val_data = random_split(training_data , [int(0.8*len(training_data)) , len(training_data) - int(0.8*len(training_data))] , generator)

  print(f"Training samples: {len(train_data)} and Validation samples:{len(val_data)}")

  train_loader = DataLoader(train_data , batch_size=16 , shuffle=True)
  val_loader = DataLoader(val_data , batch_size=16 , shuffle=False)

  loss_fn = nn.CrossEntropyLoss()
  optimizer = optim.Adam(model.parameters() , lr=0.0001)
  scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max' , factor=0.5 , patience=3)

  best_dice = 0.0 

  for epoch in range(40):
    
    print(f"Epoch: {epoch}")
    model.train()
    epoch_loss = 0.0 
    epoch_pix_acc = 0.0
    dice = 0.0 

    for idx , (images , ids , bboxs , segments) in enumerate(train_loader):
      images = images.to(device)
      segments = (segments).to(device)
      output = model(images)
      loss = loss_fn(output , segments)

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      epoch_loss += loss.item()
      predictions = torch.argmax(output , dim=1)

      correct_pred = (predictions == segments).sum().item()
      total = segments.numel()
      epoch_pix_acc += (correct_pred / total) 

      dice += dice_score(predictions , segments , 3).item()

      if (idx + 1)%20 == 0:
        print(f"Batch {idx+1}/{len(train_loader)} , Loss: {loss.item():.4f}")

    train_loss = (epoch_loss / len(train_loader))
    train_pix_acc = (epoch_pix_acc / len(train_loader)) * 100 
    train_dice = (dice / len(train_loader))
  
    print(f"Training Loss:{train_loss} ; Pixel Accuracy:{train_pix_acc} ; Dice Score:{train_dice}")

    model.eval()
    val_loss = 0.0 
    val_dice_score = 0.0 
    val_pix_acc = 0.0 

    with torch.no_grad():
      for images , ids , bboxs ,segments in val_loader:
        images = images.to(device)
        segments = segments.to(device)

        outputs = model(images)
        loss = loss_fn(outputs , segments)

        val_loss += loss.item()
        predictions = torch.argmax(outputs , dim=1)

        correct_pred = (predictions == segments).sum().item()
        total = segments.numel()
        val_pix_acc += (correct_pred / total)

        val_dice_score += dice_score(predictions , segments , 3).item()

    val_loss = (val_loss / len(val_loader))
    val_pix_acc = (val_pix_acc / len(val_loader)) * 100 
    val_dice_score = (val_dice_score / len(val_loader))

    scheduler.step(val_dice_score)

    print(f"Validation Loss:{val_loss} ; Pixel Accuracy:{val_pix_acc} ; Dice Score:{val_dice_score}")

    if val_dice_score > best_dice:
      best_dice = val_dice_score
      checkpoint = {
        "state_dict": model.state_dict() , 
        "epoch": epoch ,
        "best_metric": best_dice
      }
      torch.save(checkpoint , "./checkpoints/unet.pth")


def dice_score(predictions:torch.Tensor , ground:torch.Tensor , num_classes) -> torch.Tensor:

  dice_scores : list[torch.Tensor] = []

  for index in range(num_classes):
    # Suppose Ground truth is       [1 , 2 , 3 , 2 , 1 , 1 , 2 , 2]
    # and prediction is smthng like [1 , 1 , 3 , 1 , 2 , 2 , 2 , 3]
    # Then no. of 1 correct is 1 , 2 correct is 1 , 3 correct is 1 
    # So , for each class, we can just make it either 0 or 1 , then check. 

    pred_class = (predictions == index)
    ground_class = (ground == index)
    correct_class = (pred_class & ground_class).sum().float()
    tp_fp = pred_class.sum().float()
    tp_fn = ground_class.sum().float()

    dice = (2.0 * correct_class)/(tp_fp + tp_fn + 1e-6)

    # Dice Score is 2 x TP / (2 x TP + FP + FN)
    # Ok so for class 1 , for example , if we take the pixel for which 
    # ground is 2 and prediction is 3 , then its 0 and 0 
    # 1 and 1 = TP , 1 and 0 = FN , 0 and 1 = FP , 0 and 0 = TN 
    # 2 x TP + FP + FN = (TP + FP) + (TP + FN)
    # (TP + FP) = 1's of our prediction
    # TP + FN = 1's of ground 

    dice_scores.append(dice)
  
  return torch.stack(dice_scores).mean()



if __name__ == "__main__":
  classifier(batch_norm=True , dropout=0.5)
  localizer(batch_norm=True , dropout=0.5)
  segmentation(batch_norm=True , dropout=0.5)

