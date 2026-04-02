"""Dataset skeleton for Oxford-IIIT Pet.
"""

import os 
import numpy as np
import xml.etree.ElementTree as ET
import torch
from PIL import Image
from torch.utils.data import Dataset

class OxfordIIITPetDataset(Dataset):
    """Oxford-IIIT Pet multi-task dataset loader skeleton."""

    def __init__(self , isTrain=False , transform=None):
        # self.images = "./dataset/images"
        # self.trimaps = "./dataset/annotations/trimaps"
        # self.bounding_box = "./dataset/annotations/xmls"     

        self.transform = transform
        file = open("./data/dataset/annotations/trainval.txt")

        if not isTrain:
            file = open("./data/dataset/annotations/test.txt")
  
        self.image_set , self.image_id , self.image_bbox , self.image_segment = self.load(file)


    def load(self , inpt_file):
        image_set = []
        image_id = []
        train_species = []
        train_breedid = []
        image_segment = []
        image_bbox = []
        
        for line in inpt_file:
            inpt = line.split()
            filename = "./data/dataset/images/" + inpt[0] + ".jpg"
            segname = "./data/dataset/annotations/trimaps/" + inpt[0] + ".png"
            xmlname = "./data/dataset/annotations/xmls/" + inpt[0] + ".xml"
            id = int(inpt[1]) - 1
            species = int(inpt[2]) - 1
            breedid = int(inpt[3]) - 1

            if not (os.path.exists(filename) and os.path.exists(segname)):
                continue 

            image_id.append(id)
            train_species.append(species)
            train_breedid.append(breedid)

            # image = np.array(Image.open(filename).convert('RGB'))
            image_set.append(filename)

            # segment = np.array(Image.open(segname))
            # segment = segment - 1 
            image_segment.append(segname)

            if os.path.exists(xmlname):
                tree = ET.parse(xmlname)
                root = tree.getroot() 
                bndbox = root.find('object').find('bndbox') # type: ignore

                xmin = float(bndbox.find('xmin').text) # type: ignore
                xmax = float(bndbox.find('xmax').text) # type: ignore
                ymin = float(bndbox.find('ymin').text) # type: ignore
                ymax = float(bndbox.find('ymax').text) # type: ignore
            
            else:
                xmin = 0.0 
                ymin = 0.0 
                xmax = 1.0 
                ymax = 1.0

            image_bbox.append([xmin , ymin , xmax , ymax]) 

        return image_set , image_id , image_bbox , image_segment
    
    def __getitem__(self, index):
        image = np.array(Image.open(self.image_set[index]).convert('RGB'))
        segment = np.array(Image.open(self.image_segment[index])) 
        segment = segment - 1
        bbox = self.image_bbox[index]
        id = self.image_id[index]

        if self.transform:
            augmented = self.transform(image=image , mask=segment ,  bboxes = [bbox] , class_labels=[id]) 
            image = augmented['image']
            segment = augmented['mask']
            bbox = augmented['bboxes'][0] 

        xmin , ymin , xmax , ymax= bbox 
        x_center = (xmin + xmax) / 2.0 
        y_center = (ymin + ymax) / 2.0 
        width = xmax - xmin 
        height = ymax - ymin 

        if not isinstance(image , torch.Tensor):
            image = torch.from_numpy(image.transpose((2 , 0 , 1))).float() / 255.0 

        id_tensor = torch.tensor(id , dtype=torch.long)
        bbox_tensor = torch.tensor([x_center , y_center , width , height] , dtype = torch.float32)
        segment_tensor = torch.tensor(segment , dtype = torch.long)

        return image , id_tensor , bbox_tensor , segment_tensor 
    
    def __len__(self):
        return len(self.image_set)

        









    