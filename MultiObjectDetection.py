# -*- coding: utf-8 -*-
"""
Created on Fri Jan  8 03:40:32 2021

@author: asus
"""
from torch.utils.data import DataLoader,Dataset, Subset
from PIL import Image
import torchvision.transforms.functional as TF
import os
import numpy as np
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class CocoDataset (Dataset):
    
    def __init__(self, path2listFile, transform = None, transform_params =None):
        
        with open(path2listFile, "r") as file:
            self.path2imgs = file.readlines()
        self.path2imgs = [path.replace("/images", "./data/images") for path in self.path2imgs]

        self.path2labels = [path.replace("images", "labels").replace(".png",
                            ".txt").replace(".jpg",".txt") 
                            for path in self.path2imgs]
        self.transform_params = transform_params
        self.transform = transform
         
    def __len__(self):
        return (len(self.path2imgs))
        
    def __getitem__(self, index):
        
        path2img = self.path2imgs[index % len(self.path2imgs)].rstrip()
        img = Image.open(path2img).convert('RGB')
        path2label = self.path2labels[index % len(self.path2imgs)].rstrip()

        labels = None
        if os.path.exists(path2label):
            labels = np.loadtxt(path2label).reshape(-1,5)
        if self.transform:
            img, labels = self.transform(img, labels,
                                         self.transform_params)
        return img, labels, path2img

root_data = "./data/coco"
path2trainList  = os.path.join(root_data,"trainvalno5k.txt")
coco_train = CocoDataset(path2trainList)
print(len(coco_train))

# ===========================
img, labels, path2img = coco_train[2]
print("image size:", img.size, type(img))
print("shape of label is :",labels.shape ,type(labels))
print("labels are:", labels)

#============================
path2valList  = os.path.join(root_data,"5k.txt")
coco_val = CocoDataset(path2valList, transform=None,transform_params=None)
print(len(coco_val))

img_val, labels_val, path2img_val = coco_val[7]
print("image size:", img_val.size, type(img_val))
print("shape of label is :",labels_val.shape,type(labels_val))
print("labels are:", labels_val)

#====show sample of database
import matplotlib.pylab as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from torchvision.transforms.functional import to_pil_image
import random

path2cocNames = "./data/coco.names"
fp = open(path2cocNames,"r")
coco_names = fp.read().split("\n")[:-1]
print("number of classes:", len(coco_names))
print(coco_names)
#
def rescale_bbox(bb,W,H):
    
    x,y,w,h = bb
    return [x*W,y*H,w*W,h*H]

COLORS = np.random.randint(0,255,size = (80,3),dtype='uint8')
#fnt = ImageFont.truetype('Pillow/Tests/fonts/FreeMono.ttf',16)
fnt = ImageFont.truetype('arial.ttf',16)
COLORS = np.random.randint(0, 255, size=(80, 3),dtype="uint8")
#fnt = ImageFont.truetype('Pillow/Tests/fonts/FreeMono.ttf', 16)
fnt = ImageFont.truetype('arial.ttf',16)
def show_img_bbox(img,targets):
    if torch.is_tensor(img):
        img=to_pil_image(img)
    if torch.is_tensor(targets):
        targets=targets.numpy()[:,1:]
        
    W, H=img.size
    draw = ImageDraw.Draw(img)
    
    for tg in targets:
        id_=int(tg[0])
        bbox=tg[1:]
        bbox=rescale_bbox(bbox,W,H)
        xc,yc,w,h=bbox
        
        color = [int(c) for c in COLORS[id_]]
        name=coco_names[id_]
        
        draw.rectangle(((xc-w/2, yc-h/2), (xc+w/2, yc+h/2)),outline=tuple(color),width=3)
        draw.text((xc-w/2,yc-h/2),name, font=fnt, fill=(255,255,255,0))
    plt.imshow(np.array(img)) 
#show a sample image from coco_train
np.random.seed(21)
rnd_ind = np.random.randint(len(coco_train))
img, labels , path2img = coco_train[rnd_ind]
print(img.size)
plt.rcParams['figure.figsize'] = (20,10)
show_img_bbox(img, labels)

#show a sample image from coco_val
np.random.seed(0)
rnd_ind = np.random.randint(len(coco_val))
img, labels , path2img = coco_val[rnd_ind]
print(img.size, labels.shape)
plt.rcParams['figure.figsize'] = (20,10)
show_img_bbox(img, labels)
#================================Transforming data
def pad_to_squere(img,boxes, pad_value=0, normalized_labels = True):
              
    
    w,h = img.size
    w_factor, h_factor = (w,h) if normalized_labels else (1,1)
    dim_diff = np.abs(h-w)
    pad1 = dim_diff // 2
    pad2 = dim_diff - pad1
    
    if h<w :
        left, top, right, bottom = 0,pad1,0,pad2
    else:
        left, top, right, bottom = pad1 ,0, pad2,0
    padding = (left, top, right, bottom)
    img_padded = TF.pad(img, padding=padding, fill=pad_value)
    w_padded, h_padded = img_padded.size
    
    # calculating the coordinates of the top left of the bounding box and 
    x1 = w_factor * (boxes[:,1] - boxes[:,3]/2)
    y1 = h_factor * (boxes[:,2] - boxes[:,4]/2) 
    # calculating the coordinates of the bottom left of the bounding box
    x2 = w_factor * (boxes[:,1] + boxes[:,3]/2)
    y2 = h_factor * (boxes[:,2] + boxes[:,4]/2)  
    
    x1 += padding[0]
    y1 += padding[1]    
    x2 += padding[2] 
    y2 += padding[3]
    
    # calculating the coordinates of the centroid of the bounding box and normalizing it
    boxes[:, 1] = ((x1+x2)/2) /w_padded
    boxes[:, 2] = ((y1+y2)/2) /h_padded
    # befor width and hight were normalized with width and hieght of image(width/w_factor and hight/h_factor)
    # here, width and hight are normalized with width and hieght of image that are padded(width/w_padded and hight/h_padded)
    boxes[:, 3] *=  w_factor  / w_padded
    boxes[:, 4] *=  h_factor /  h_padded
    
    return img_padded, boxes

def hflip(img, labels):
    image = TF.hflip(img)
    labels[:,1] = 1.0 - labels[:,1]
    return image, labels

def transformer(image, labels,params):
    
    if params["pad2square"] is True:
        image, labels = pad_to_squere(image, labels)
    image = TF.resize(image, params["target_size"])
    if random.random() < params["p_hflip"]:
         image, labels = hflip(image, labels)
    image = TF.to_tensor(image)
    targets = torch.zeros((len(labels),6))
    targets[:,1:]=torch.from_numpy(labels)
    return image, targets


trans_params_train = {
                        "target_size" : (416,416),
                        "pad2square" : True,
                        "p_hflip" : 1.0,
                        "normalized_labels": True,
                        }

coco_train = CocoDataset(path2trainList,
                         transform=transformer,
                         transform_params=trans_params_train)
np.random.seed(21)
rnd_ind = np.random.randint(len(coco_train))
img, targets, path2img = coco_train[rnd_ind]
print("image shape:", img.shape)
print("target shape:", targets.shape)

plt.rcParams['figure.figsize'] = (20,10)
COLORS = np.random.randint(0,255,size = (80,3), dtype="uint8")
show_img_bbox(img, targets)

#===

trans_params_val = {
                        "target_size" : (416,416),
                        "pad2square" : True,
                        "p_hflip" : 0.0,
                        "normalized_labels": True,
                        }

coco_val = CocoDataset(path2valList,
                         transform=transformer,
                         transform_params=trans_params_val)
np.random.seed(0)
rnd_ind = np.random.randint(len(coco_val))
img, targets, path2img = coco_val[rnd_ind]
print("image shape:", img.shape)
print("target shape:", targets.shape)

plt.rcParams['figure.figsize'] = (20,10)
COLORS = np.random.randint(0,255,size = (80,3), dtype="uint8")
show_img_bbox(img, targets)

#==============================Defining the Dataloaders

def collate_fn(batch):
    
    #Group the images, targets, and paths in the mini-batch
    imgs,targets,paths = list(zip(*batch))
    #Remove empty boxes
    targets = [boxs for boxs in targets if boxs is not None]
    #Set the sample index
    for b_i, boxes in enumerate(targets):
        boxes[:,0] = b_i
    targets = torch.cat(targets,0)
    #Convert list to tensor
    imgs = torch.stack([img for img in imgs])
    return imgs, targets, paths


batch_size = 8
train_dl = DataLoader(
    coco_train,
    batch_size=batch_size,
    shuffle=True,
    num_workers = 0,
    pin_memory =True,
    collate_fn= collate_fn,   
    )

torch.manual_seed(0)

for imgs_batch, tg_batch, path_batch in train_dl:
    print(tg_batch[2,0])
    break

print(imgs_batch.shape)
print(tg_batch.shape, tg_batch.dtype)

#=====

val_dl = DataLoader(
    coco_val,
    batch_size=batch_size,
    shuffle=False,
    num_workers = 0,
    pin_memory =True,
    collate_fn= collate_fn,   
    )

torch.manual_seed(0)
for imgs_batch, tg_batch, path_batch in val_dl:
    break

print(imgs_batch.shape)
print(tg_batch.shape, tg_batch.dtype)    

    
#======================Parsing the configuration file
from myutils import parse_model_config

path2config = "./config/yolov3.cfg"
blocks_list = parse_model_config(path2config)
blocks_list
  
    
    
    
    
    

        
        
        
        
        
    




    
            
            
            
            
        