#dataloader for pre-training
import torch
import torchvision 
import torch.nn as nn 
import torch.nn.functional as F 
from torchvision    import datasets 
from torchvision import transforms 
from torchvision.utils import save_image 
import torch.utils.data as data 
import numpy as np
from PIL import ImageEnhance,Image
import random
import os

def norm(img):
    img -= img.min(1, keepdim=True)[0]
    img /= img.max(1, keepdim=True)[0]
    return img

def cv_random_flip(img):
    # left right flip
    flip_flag = random.randint(0, 2)
    if flip_flag == 1:
        img = np.flip(img, 0).copy()
    if flip_flag == 2:
        img = np.flip(img, 1).copy()
    return img

def randomRotation(image):
    rotate_time = random.randint(0, 3)
    image = np.rot90(image, rotate_time).copy()
    return image

def colorEnhance(image):
    bright_intensity = random.randint(8, 12) / 10.0
    image = ImageEnhance.Brightness(image).enhance(bright_intensity)
    contrast_intensity = random.randint(8, 12) / 10.0
    image = ImageEnhance.Contrast(image).enhance(contrast_intensity)
    color_intensity = random.randint(0, 20) / 10.0
    image = ImageEnhance.Color(image).enhance(color_intensity)
    sharp_intensity = random.randint(0, 30) / 10.0
    image = ImageEnhance.Sharpness(image).enhance(sharp_intensity)
    return image

def randomGaussian(img, mean=0.002, sigma=0.002):

    def gaussianNoisy(im, mean=mean, sigma=sigma):
        for _i in range(len(im)):
            im[_i] += random.gauss(mean, sigma)
        return im

    flag = random.randint(0, 3)
    if flag == 1:
        width, height = img.shape
        img = gaussianNoisy(img[:].flatten(), mean, sigma)
        img = img.reshape([width, height])

    return img


def randomPeper(img):
    flag = random.randint(0, 3)
    if flag == 1:
        noiseNum = int(0.0015 * img.shape[0] * img.shape[1])
        for i in range(noiseNum):
            randX = random.randint(0, img.shape[0] - 1)
            randY = random.randint(0, img.shape[1] - 1)
            if random.randint(0, 1) == 0:
                img[randX, randY] = 0
            else:
                img[randX, randY] = 1
    return img

class MAE_Dataset(data.Dataset):
    def __init__(self,img_size,image_root,modality,augment=False):

        self.modal_list = ['t1', 't2', 't1ce', 'flair', 'gt']
        self.image_root = image_root
        self.modality = modality
        self.images = [self.image_root + f for f in os.listdir(self.image_root) if f.endswith('.npy')]
        self.images.sort(key=lambda x: int(x.split(image_root)[1].split(".npy")[0]))
        self.img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(img_size)
        ])
        self.gt_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(img_size,Image.NEAREST)
        ])
        self.Len = int(len(self.images))
        self.augment = augment
        print('slice number:',self.__len__())

    def __getitem__(self, index):
        if self.modality == 'all':
            modal = int(index / self.Len)
            subject = int(index % self.Len)
            npy = np.load(self.images[subject])
            img = npy[modal, :, :]
        else:
            modal = self.modal_list.index(self.modality)
            npy = np.load(self.images[index])
            img = npy[modal, :, :]

        if self.augment == True:
            img = cv_random_flip(img)
            img = randomRotation(img)
            img = randomGaussian(img)
            img = randomPeper(img)
            img = img * 255
            img = Image.fromarray(img.astype(np.uint8))
            img = colorEnhance(img)
            img = img.convert('L')
        img = self.img_transform(img)
        return img

    def __len__(self):
        if self.modality == 'all':
            return int(len(self.images))* 4
        else:
            return int(len(self.images))

def get_maeloader(batchsize, shuffle,modality,pin_memory=True,source_modal='t1', target_modal='t2',
        img_size = 256,img_root='../data/train/',num_workers=16,augment=False):
    dataset = MAE_Dataset(img_size=img_size,image_root=img_root,augment=augment,modality=modality)
    data_loader = data.DataLoader(dataset=dataset, batch_size=batchsize, shuffle=shuffle,pin_memory=pin_memory,num_workers=num_workers)
    return data_loader
