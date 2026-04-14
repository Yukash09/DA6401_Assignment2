"""Inference and evaluation
"""

import torch
import torch.nn as nn
import torchvision.utils
from torch.utils.data import DataLoader , random_split
import torch.optim as optim
from PIL import Image
# import copy
import gc
import albumentations as A
import wandb
import numpy as np
from albumentations.pytorch import ToTensorV2 
from data.pets_dataset import OxfordIIITPetDataset
# from models.classification import VGG11Classifier as VGGC
# from models.localization import VGG11Localizer as VGGL
# from models.segmentation import VGG11UNet as VGGU
from models.multitask import MultiTaskPerceptionModel as MultiTask
# from losses.iou_loss import IoULoss

def mappesh():
    file1 = open("./data/dataset/annotations/list.txt")
    mapp = {}

    for line in file1:
        if line.startswith('#'):
            continue 

        arr = line.split()
        name =  arr[0]
        ids = int(arr[1]) - 1
        onlyname = name.split('_') 
        fullname = ""
        for i in range(len(onlyname) - 1):
            fullname += onlyname[i]

        mapp[ids] = fullname 
    
    return mapp 

def inference():
    wandb.init(project="Fun")
    mapp = mappesh()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MultiTask(
        num_breeds=37,
        seg_classes=3,
        in_channels=3, 
        classifier_path="./checkpoints/classifier.pth",
        localizer_path="./checkpoints/localizer.pth",
        unet_path="./checkpoints/unet.pth",
    ).to(device) 

    model.eval()

    transform = A.Compose([
        A.Resize(224 , 224),
        A.Normalize(),
        ToTensorV2()
    ])

    columns = ["Image Name" , "Output" , "Prediction" , "Confidence"]
    table = wandb.Table(columns)

    # Pre-downloaded from wikipaedia and some random site through google images section search bar
    img1 = "./data/q7/1000203995.jpg"
    img2 = "./data/q7/1000203996.jpg"
    img3 = "./data/q7/1000203997.jpg"

    with torch.no_grad():
        wbimg1 , conf1 , pred1 = inf_help(img1 , transform , device , model , mapp) 
        wbimg2 , conf2 , pred2 = inf_help(img2 , transform , device , model , mapp) 
        wbimg3 , conf3 , pred3 = inf_help(img3 , transform , device , model , mapp) 

        table.add_data("Samoyed" , wbimg1 , pred1 , conf1.item())
        table.add_data("Saint Bernard" , wbimg2 , pred2 , conf2.item())
        table.add_data("Basset Hound" , wbimg3 , pred3 , conf3.item())

    wandb.log({"Final Pipeline Showcase": table})
    wandb.finish()


def inf_help(img , transform , device , model , mapp):
    act_img = Image.open(img).convert('RGB').resize((224 , 224))
    base_array = np.array(act_img)

    inpt = transform(image=base_array)['image'].unsqueeze(0).to(device)

    output = model(inpt)

    logits = output['classification']
    bboxs = output['localization'][0].cpu().numpy()
    seg_logits = output['segmentation']

    confidence , preds = torch.max(torch.softmax(logits , dim=1) , dim=1)
    cx , cy , w , h = bboxs[0] , bboxs[1] , bboxs[2] ,  bboxs[3]
    seg_preds = torch.argmax(seg_logits , dim=1)[0].cpu().numpy()

    pred_name = mapp.get(preds.item() , f"Class {preds.item()}")

    def func(cx , cy , w, h):
        return {
        "minX": float(cx - w/2)/224.0 , 
        "maxX": float(cx + w/2)/224.0 ,
        "minY": float(cy - h/2)/224.0 ,
        "maxY": float(cy + h/2)/224.0
        }

    wb_img = wandb.Image(
        base_array,
        boxes={
            "predictions":{
                "box_data":[
                    {
                        "position": func(cx , cy , w , h) , 
                        "class_id": preds.item() , 
                        "box_caption": pred_name
                    }
                ] , 
                "class_labels":{preds.item(): pred_name}
            }
        },
        masks={
            "predictions":{
                "mask_data": seg_preds,
                "class_labels":{
                    0: "Foreground" ,
                    1: "Background" ,
                    2: "Not Classified"
                }
            }
        }
    )

    return wb_img , confidence , pred_name

if __name__ == "__main__":
    inference()