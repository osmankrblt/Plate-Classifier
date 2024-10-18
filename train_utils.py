import os
from typing import Optional
import cv2
import numpy as np
import pandas as pd
import torch
from torchvision.transforms import v2
from torch import nn as nn
from torch.nn import Conv2d, MaxPool2d, BatchNorm2d, Linear
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.io import read_image
import random
import torchvision
import math


def getTransformFunc(image):

    optional_transforms = [
            v2.GaussianNoise(), 
            
            #v2.RandomAdjustSharpness(sharpness_factor=2.0, p=0.5), 
            #v2.RandomAutocontrast(p=0.5), 
            #v2.RandomEqualize(),
            #v2.RandomCrop((256,256)),
            #v2.RandomAffine(degrees=0, translate=(0.1,0.1), scale=(0.9,1.1), shear=0, fill=0),
            #v2.RandomPerspective(distortion_scale=0.3, p=0.5),
        
            #v2.ColorJitter(brightness=0.5),
            
            #v2.RandomRotation(10),
            v2.GaussianBlur(kernel_size=3)
        ]

    return v2.Compose([
        #v2.GaussianNoise(), 
        #v2.RandomEqualize(),
        #v2.ColorJitter(brightness=0.2),
        v2.RandomAutocontrast(p=0.5), 
        #v2.GaussianBlur(kernel_size=3),
        #v2.RandomHorizontalFlip(),
        #v2.RandomVerticalFlip(),
        v2.CenterCrop((256,256)),
        v2.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5]),

     ])(image)

def create_circular_mask_from_tensor(input_tensor, ratio):
    """
    PyTorch tensoru alarak görüntünün ortasında belirtilen oranda yuvarlak bir alan oluşturur
    ve geri kalan alanı siyah yapar.

    Args:
        input_tensor (torch.Tensor): Giriş görüntüsünün tensoru (C, H, W formatında).
        ratio (float): Yuvarlak alanın oranı (0.0 ile 1.0 arasında).

    Returns:
        torch.Tensor: Yuvarlak alan içeren ve geri kalan alanı siyah olan görüntü.
    """
    # Tensoru numpy dizisine çevirme (C, H, W) formatından (H, W, C) formatına
    img = input_tensor.permute(1, 2, 0).numpy()
    height, width = img.shape[:2]

    # Yuvarlak alanın merkezi ve yarıçapı
    center = (width // 2, height // 2)
    radius = int(min(width, height) * ratio / 2)

    # Maske oluşturma
    mask = np.zeros((height, width), dtype=np.uint8)

    # Yuvarlak alanı beyaz ile doldurma
    cv2.circle(mask, center, radius, 255, -1)

    # Orijinal görüntüyü siyah arka plana yerleştirme
    result = np.zeros_like(img)  # Siyah görüntü
    result[mask == 255] = img[mask == 255]  # Yuvarlak alandaki renkleri koru

    # Sonucu yeniden tensor formatına çevirme (H, W, C) formatından (C, H, W) formatına
    result_tensor = torch.from_numpy(result).permute(2, 0, 1)  # C, H, W formatına çevir

    return result_tensor

# 2. Merkezi zoom işlemi
def zoom_center(image, zoom_factor):

    _, height, width = image.shape
    new_height = math.floor(height / zoom_factor)  # Tam sayıya yuvarlama
    new_width = math.floor(width / zoom_factor)    # Tam sayıya yuvarlama
    
    # Merkezden kırpma başlangıç koordinatları
    start_x = (width - new_width) // 2
    start_y = (height - new_height) // 2
    
    # Resmi kırp ve yeniden boyutlandır
    cropped_image = image[:, start_y:start_y + new_height, start_x:start_x + new_width]
    
    # Boyutlandırmayı orijinal boyuta döndür
    resized_image = torch.nn.functional.interpolate(cropped_image.unsqueeze(0), size=(height, width), mode='bilinear', align_corners=False).squeeze(0)
    
    return resized_image



import torch
import torch.nn as nn

class PlateModel(nn.Module):
        
    def __init__(self, img_size=256):
        super(PlateModel, self).__init__()

        self.img_size = img_size

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)  # Batch Normalization katmanı
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)  # Batch Normalization katmanı
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(256)  # Batch Normalization katmanı
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(256 * (self.img_size // 8) * (self.img_size // 8), 128)
        
        self.dropout2 = nn.Dropout(0.1)
        self.fc2 = nn.Linear(128, 2)  # İki sınıf için çıkış katmanı

    def forward(self, x):
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))  
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))  
        x = self.pool(torch.relu(self.bn3(self.conv3(x))))  
        
        x = x.view(-1, 256 * (self.img_size // 8) * (self.img_size // 8))  
        x = torch.relu(self.dropout(x))
        x = self.fc1(x)
        x = torch.relu(self.dropout2(x)) 
        x = self.fc2(x)
        
        return x


    



class PlateDataset(Dataset):

        def __init__(self, dataset_path:str, mode:str, csv_file_path:Optional[str]=None,  transform:bool=False):
            super().__init__()

            self.label2id = {"dirty":0,"cleaned":1}
            self.id2label = {"0":"dirty","1":"cleaned"}
            self.transfrom = transform
            
            self.mode = mode

            if self.mode=="test":
                # Resimleri topla
                self.image_paths = [os.path.join(dataset_path,x) for x in os.listdir(dataset_path) if x.endswith(".jpg")]
                # CSV dosyasını oku
                self.df = pd.read_csv(csv_file_path)
                

            else:

                self.image_paths = [os.path.join(os.path.join(dataset_path, label),x)   for label in ["cleaned","dirty"] for x in os.listdir(os.path.join(dataset_path, label)) if x.endswith(".jpg")]

            
                
            
        def __len__(self):

            return len(self.image_paths) 

        
        def __getitem__(self, index):
            

            if self.mode=="test":
                img_id = self.image_paths[index].split("/")[-1].split(".")[0].strip()
                
                
                # CSV dosyasından resmin sınıfını getir ve id'ye çevir
                img_label = self.label2id[self.df.loc[self.df['id'] == int(img_id), "label"].values[0]]

            else:
                img_label = self.label2id[self.image_paths[index].split("/")[-2]]

           

            img = read_image(self.image_paths[index]).to(dtype=torch.float32)

            img = v2.Resize((300, 300))(img)


            #zoom_factor = 2  # Yakınlaştırma faktörü
            #img = zoom_center(img, zoom_factor)
            
            img = create_circular_mask_from_tensor(img, 0.4)

            if self.transfrom==True:

                img = getTransformFunc(img)

            elif self.transfrom==False:
                img = v2.Resize((256, 256))(img)
                img = v2.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])(img)
               
            return torch.tensor(img, dtype=torch.float32), torch.tensor(img_label, dtype=torch.float32)
            


def getDataLoaders(train_dataset: Dataset, test_dataset:Dataset, shuffle: bool=True, batch_size: int=1):
     


    # DataLoader'ları oluştur
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    test_loader = DataLoader(test_dataset, batch_size=batch_size*5, shuffle=shuffle)
        


    return train_loader, test_loader
    
    

        

    