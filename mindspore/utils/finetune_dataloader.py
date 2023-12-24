import numpy as np
import argparse
from mindspore.dataset import MnistDataset, GeneratorDataset
from mindspore.dataset import transforms, vision
import mindspore.dataset.transforms.c_transforms as C
import random
from PIL import ImageEnhance,Image
import os
from einops import rearrange
import scipy.ndimage as ndimage

def patchify(imgs,patch_size = 8):
    imgs = imgs.asnumpy()
    p = patch_size
    assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0
    h = w = imgs.shape[2] // p
    x = rearrange(imgs, 'n c (h p1) (w p2) -> n h w (p1 p2) c', p1=p, p2=p)
    x = rearrange(x, 'n h w (p1 p2) c -> n (h w) (p1 p2 c)', p1=p, p2=p)
    return x

def unpatchify(x,patch_size = 8):
    x = x.asnumpy()
    p = patch_size
    h = w = int(x.shape[1] ** 0.5)
    assert h * w == x.shape[1]
    x = rearrange(x, 'n (h w) (p1 p2) c -> n h w p1 p2 c', h=h, w=w, p1=p, p2=p)
    x = rearrange(x, 'n h w p1 p2 c -> n c h p1 w p2', p1=p, p2=p)
    imgs = rearrange(x, 'n c h p1 w p2 -> n c (h p1) (w p2)', p1=p, p2=p)
    return imgs

def cv_random_flip(img):
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

class Finetune_Dataset:

    def __init__(self,img_size,image_root,source_modality,target_modality,augment):
        self.modal_list = ['t1', 't2', 't1ce', 'flair', 'gt']
        self.image_root = image_root
        self.source_modality = source_modality
        self.target_modality = target_modality
        self.augment = augment
        self.img_size = img_size
        self.images = [self.image_root + f for f in os.listdir(self.image_root) if f.endswith('.npy')]
        self.images.sort(key=lambda x: int(x.split(image_root)[1].split(".npy")[0]))
        self.Len = int(len(self.images))
        self.resize = vision.Resize(self.img_size)
        self.ToTensor = vision.ToTensor()
        self.HWC2CHW = vision.HWC2CHW()
        print('slice number:',self.__len__())

    def __getitem__(self, index):

        npy = np.load(self.images[index])
        source_modal = self.modal_list.index(self.source_modality)
        source_img = npy[source_modal, :, :]
        target_modal = self.modal_list.index(self.target_modality)
        target_img = npy[target_modal, :, :]

        if self.augment == True:
            source_img, target_img = cv_random_flip(source_img), cv_random_flip(target_img)
            source_img, target_img = randomRotation(source_img), randomRotation(target_img)
            source_img, target_img = randomGaussian(source_img), randomGaussian(target_img)
            source_img, target_img = randomPeper(source_img), randomPeper(target_img)
        source_img, target_img = source_img * 255, target_img * 255
        source_img, target_img = Image.fromarray(source_img.astype(np.uint8)), Image.fromarray(target_img.astype(np.uint8))
        source_img, target_img = colorEnhance(source_img), colorEnhance(target_img)
        source_img, target_img = source_img.convert('L').resize(self.img_size), target_img.convert('L').resize(self.img_size)
        source_img, target_img = self.ToTensor(source_img), self.ToTensor(target_img)

        return source_img, target_img

    def __len__(self):
        return int(len(self.images))

if __name__ == '__main__':
    loader = Finetune_Dataset(img_size=(256, 256), image_root='../../data/train/', source_modality='t1',target_modality='t2')
    dataset = GeneratorDataset(source=loader, column_names=['source', 'target'])
    dataset = dataset.batch(batch_size = 10)
    for batch in dataset.create_dict_iterator():
        source_img, target_img = batch['source'], batch['target']
        print(source_img.shape, target_img.shape)