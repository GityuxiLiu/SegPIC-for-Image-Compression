# Copyright 2020 InterDigital Communications, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset

import torch
import torch.nn.functional as F
import random
import numpy as np
import os
import torchvision.transforms as transforms
from panopticapi.utils import rgb2id
import json

# Change down_type for Pytorch-2.0
# down_type = Image.Resampling.LANCZOS
down_type = Image.ANTIALIAS

class ImageFolder(Dataset):
    def __init__(self, root, transform=None, split="train", noAugment=False, p_aug=0):
        if split == "train":
            splitdir = Path(root) / "train2017"
        elif split == "test":
            splitdir = Path(root) / "val2017"
        self.split = split
        self.mask_root = os.path.join(root, "annotations/panoptic_train2017/")
        self.mask_root_test = os.path.join(root, "annotations/panoptic_val2017/")
        
        self.noAugment = noAugment

        if not splitdir.is_dir():
            raise RuntimeError(f'Invalid directory "{root}"')

        self.samples = [f for f in splitdir.iterdir() if f.is_file()]
        self.transform = transform
        self.p_aug = p_aug

    def __getitem__(self, index):
        img_path = self.samples[index]
        img_name = os.path.basename(img_path)
        img_name = os.path.splitext(img_name)[0] + ".png"

        img = Image.open(img_path).convert("RGB")
        if self.split == "train":
            mask = Image.open(os.path.join(self.mask_root, img_name)).convert("RGB")
        elif self.split == "test":
            mask = Image.open(os.path.join(self.mask_root_test, img_name)).convert("RGB")
        
        width, height = img.size
        assert img.size == mask.size, "the img dismatch mask !"
        if width<256 or height<256:
            img = resize256(img)
            mask = resize256(mask,True)

        # focus on global or local randomly 
        elif random.random()<self.p_aug and not self.noAugment and self.split=="train":
            if width < height:
                img = img.resize((256, height*256//width), down_type)
                mask = mask.resize((256, height*256//width), 0)
            else:
                img = img.resize((width*256//height, 256), down_type)
                mask = mask.resize((width*256//height, 256), 0)
        assert img.size == mask.size, "the img dismatch mask !"
        transform1 = transforms.Compose([transforms.ToTensor()])
        img = transform1(img)
        mask = transform1(mask)

        if self.transform:
            combine = self.transform(torch.cat([img, mask],0))
            img = combine[:3,:,:]
            mask = combine[3:,:,:]
        mask,_ = mask_rgb2gray(mask)
        
        return img, mask

    def __len__(self):
        return len(self.samples)

class ImageFolder_nomask(Dataset):
    def __init__(self, root, transform=None, split="train", noAugment=False, p_aug=0):
        if split == "train":
            splitdir = Path(root) / "train2017"
        elif split == "test":
            splitdir = Path(root) / "val2017"

        self.split = split

        if not splitdir.is_dir():
            raise RuntimeError(f'Invalid directory "{root}"')

        self.samples = [f for f in splitdir.iterdir() if f.is_file()]
        self.transform = transform
        self.noAugment = noAugment
        self.p_aug = p_aug

    def __getitem__(self, index):
        img_path = self.samples[index]
        img_name = os.path.basename(img_path)
        img_name = os.path.splitext(img_name)[0] + ".png"

        img = Image.open(img_path).convert("RGB")
        
        width, height = img.size
        if width<256 or height<256:
            img = resize256(img, size=256)
        # focus on global or local randomly 
        elif random.random()<self.p_aug and not self.noAugment and self.split=="train":
            if width < height:
                img = img.resize((256, height*256//width), down_type)
            else:
                img = img.resize((width*256//height, 256), down_type)
        transform1 = transforms.Compose([transforms.ToTensor()])
        img = transform1(img)
        if self.transform:
            img = self.transform(img)
        return img

    def __len__(self):
        return len(self.samples)


def resize256(im, ifmask=False, size=256):
    if isinstance(im,torch.Tensor):
        width, height = im.shape[-2:]
        width = max(size, width)
        height = max(size, height)
        return transforms.Resize(width, height)(im)
    else:
        width, height = im.size
        width = max(size, width)
        height = max(size, height)
        if ifmask:
            return im.resize((width, height), 0)
        # return im.resize((width, height), Image.ANTIALIAS)
        return im.resize((width, height), down_type)

def tensor2img(tensor, path, ifend=False):
    loader = transforms.Compose([transforms.ToTensor()])  
    unloader = transforms.ToPILImage()
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    image = unloader(image)
    image.save(path)

# mask_rgb(3,W,H): 3 channel rgb mask
# id(W,H): 1 channel mask with id
# area(W*H): vector, area(i) = pixel nums of (mask_id==i)
def mask_rgb2gray(mask_rgb):
    mask = mask_rgb[0,:,:]*1e6 + mask_rgb[1,:,:]*1e3 + mask_rgb[2,:,:]
    width, height = mask.size()
    mask = torch.reshape(mask,(1,width*height)).squeeze(0)
    out,id,area = torch.unique(mask, return_inverse=True, return_counts=True, dim=0)
    id = torch.reshape(id, (width, height))
    # print("len(area):",len(area))
    return id, area

# gray_expand(1,H,W)
def Mask_gray(img, mask):
    R = img[0,:,:]
    G = img[1,:,:]
    B = img[2,:,:]
    gray = 0.299*R + 0.587*G + 0.114*B #(H,W)
    mask_oh = F.one_hot(mask) 
    mask_pm = mask_oh.permute(2,0,1) #(num_mask,H,W)
    gray = gray.unsqueeze(0) #(1,H,W)

    area = torch.sum(mask_pm, dim=(-2,-1)) #(num_mask)
    gray_center = torch.sum(gray*mask_pm, dim=(-2,-1))/area #(num_mask)
    gray_center = gray_center.unsqueeze(1).unsqueeze(2) #(num_mask,H,W)

    gray_expand = torch.sum((gray_center*mask_pm),dim=0)
    gray_expand = gray_expand.unsqueeze(0)
    return gray_expand

def grid_mask(w, h, grid=64):
    down_step_w = h//grid
    down_step_h = w//grid
    x = torch.ones(h//down_step_w, w//down_step_h)
    for j in range(down_step_h):
        x_sum = torch.zeros_like(x)
        for i in range(down_step_w-1):
            y = (i+1)*x
            x_sum = torch.cat((x_sum,y),dim=1)
        if j==0:
            xx_sum = x_sum
        else:
            xx_sum = torch.cat((xx_sum, x_sum+down_step_w*j),dim=0)
    xx_sum = xx_sum.to(torch.int64)
    return xx_sum.contiguous()

    

