# dataloader for fine-tuning
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets
from torchvision import transforms
from torchvision.utils import save_image
import torch.utils.data as data
import numpy as np
from PIL import ImageEnhance, Image
import random
import os

def cv_random_flip(img, label):
    # left right flip
    flip_flag = random.randint(0, 2)
    if flip_flag == 1:
        img = np.flip(img, 0).copy()
        label = np.flip(label, 0).copy()
    if flip_flag == 2:
        img = np.flip(img, 1).copy()
        label = np.flip(label, 1).copy()
    return img, label

def randomCrop(image, label):
    border = 30
    image_width = image.size[0]
    image_height = image.size[1]
    crop_win_width = np.random.randint(image_width - border, image_width)
    crop_win_height = np.random.randint(image_height - border, image_height)
    random_region = (
        (image_width - crop_win_width) >> 1, (image_height - crop_win_height) >> 1, (image_width + crop_win_width) >> 1,
        (image_height + crop_win_height) >> 1)
    return image.crop(random_region), label.crop(random_region)

def randomRotation(image, label):
    rotate_time = random.randint(0, 3)
    image = np.rot90(image, rotate_time).copy()
    label = np.rot90(label, rotate_time).copy()
    return image, label

def colorEnhance(image):
    bright_intensity = random.randint(7, 13) / 10.0
    image = ImageEnhance.Brightness(image).enhance(bright_intensity)
    contrast_intensity = random.randint(4, 11) / 10.0
    image = ImageEnhance.Contrast(image).enhance(contrast_intensity)
    color_intensity = random.randint(7, 13) / 10.0
    image = ImageEnhance.Color(image).enhance(color_intensity)
    sharp_intensity = random.randint(7, 13) / 10.0
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


class BraTS_Train_Dataset(data.Dataset):
    def __init__(self, source_modal, target_modal, img_size,
                 image_root, data_rate, sort=False, argument=False, random=False):

        self.source = source_modal
        self.target = target_modal
        self.modal_list = ['t1', 't2', 't1ce', 'flair', 'gt']
        self.image_root = image_root
        self.data_rate = data_rate
        self.images = [self.image_root + f for f in os.listdir(self.image_root) if f.endswith('.npy')]
        self.images.sort(key=lambda x: int(x.split(image_root)[1].split(".npy")[0]))
        self.img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(img_size)
        ])
        self.gt_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(img_size, Image.NEAREST)
        ])
        self.sort = sort
        self.argument = argument
        self.random = random
        self.subject_num = len(self.images) // 60
        if self.random == True:
            subject = np.arange(self.subject_num)
            np.random.shuffle(subject)
            self.LUT = []
            for i in subject:
                for j in range(60):
                    self.LUT.append(i * 60 + j)
        print('slice number:', self.__len__())

    def __getitem__(self, index):
        if self.random == True:
            index = self.LUT[index]
        npy = np.load(self.images[index])
        img = npy[self.modal_list.index(self.source), :, :]
        gt = npy[self.modal_list.index(self.target), :, :]
        
        if self.argument == True:
            img, gt = cv_random_flip(img, gt)
            img, gt = randomRotation(img, gt)
            img = img * 255
            img = Image.fromarray(img.astype(np.uint8))
            img = colorEnhance(img)
            img = img.convert('L')

        img = self.img_transform(img)
        gt = self.img_transform(gt)
        return img, gt

    def __len__(self):
        return int(len(self.images) * self.data_rate)

def get_loader(batchsize, shuffle, pin_memory=True, source_modal='t1', target_modal='t2',
               img_size=256, img_root='../data/train/', data_rate=0.1, num_workers=8, sort=False, argument=False,
               random=False):
    dataset = BraTS_Train_Dataset(source_modal=source_modal, target_modal=target_modal,
                                  img_size=img_size, image_root=img_root, data_rate=data_rate, sort=sort,
                                  argument=argument, random=random)
    data_loader = data.DataLoader(dataset=dataset, batch_size=batchsize, shuffle=shuffle,
                                  pin_memory=pin_memory, num_workers=num_workers)
    return data_loader
