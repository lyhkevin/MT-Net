import numpy as np
from matplotlib import pylab as plt
import nibabel as nib
import random
import glob
import os
from PIL import Image
import imageio

def normalize(image, mask=None, percentile_lower=0.2, percentile_upper=99.8):

    if mask is None:
        mask = image != image[0, 0, 0]
    cut_off_lower = np.percentile(image[mask != 0].ravel(), percentile_lower)
    cut_off_upper = np.percentile(image[mask != 0].ravel(), percentile_upper)
    res = np.copy(image)
    res[(res < cut_off_lower) & (mask != 0)] = cut_off_lower
    res[(res > cut_off_upper) & (mask != 0)] = cut_off_upper
    res = res / res.max()  # 0-1

    return res

def visualize(t1_data,t2_data,flair_data,t1ce_data,gt_data):

    plt.figure(figsize=(8, 8))
    plt.subplot(231)
    plt.imshow(t1_data[:, :], cmap='gray')
    plt.title('Image t1')
    plt.subplot(232)
    plt.imshow(t2_data[:, :], cmap='gray')
    plt.title('Image t2')
    plt.subplot(233)
    plt.imshow(flair_data[:, :], cmap='gray')
    plt.title('Image flair')
    plt.subplot(234)
    plt.imshow(t1ce_data[:, :], cmap='gray')
    plt.title('Image t1ce')
    plt.subplot(235)
    plt.imshow(gt_data[:, :])
    plt.title('GT')
    plt.show()

def visualize_to_gif(t1_data, t2_data, t1ce_data, flair_data):
    transversal = []
    coronal = []
    sagittal = []
    slice_num = t1_data.shape[2]
    for i in range(slice_num):
        sagittal_plane = np.concatenate((t1_data[:, :, i], t2_data[:, :, i],
                              t1ce_data[:, :, i],flair_data[:, :, i]),axis=1)
        coronal_plane = np.concatenate((t1_data[i, :, :], t2_data[i, :, :],
                              t1ce_data[i, :, :],flair_data[i, :, :]),axis=1)
        transversal_plane = np.concatenate((t1_data[:, i, :], t2_data[:, i, :],
                              t1ce_data[:, i, :],flair_data[:, i, :]),axis=1)
        transversal.append(transversal_plane)
        coronal.append(coronal_plane)
        sagittal.append(sagittal_plane)
    imageio.mimsave("./transversal_plane.gif", transversal, duration=0.01)
    imageio.mimsave("./coronal_plane.gif", coronal, duration=0.01)
    imageio.mimsave("./sagittal_plane.gif", sagittal, duration=0.01)
    return

if __name__ == '__main__':

    t1_list = sorted(glob.glob(
        '../data/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/*/*t1.*'))
    t2_list = sorted(glob.glob(
        '../data/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/*/*t2.*'))
    t1ce_list = sorted(glob.glob(
        '../data/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/*/*t1ce.*'))
    flair_list = sorted(glob.glob(
        '../data/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/*/*flair.*'))
    gt_list = sorted(glob.glob(
        '../data/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/*/*seg.*'))

    data_len = len(gt_list) 
    train_len = int(data_len * 0.8)
    test_len = data_len - train_len
    
    train_path = '../data/train/'
    test_path = '../data/test/'

    os.makedirs(train_path,exist_ok=True)
    os.makedirs(test_path,exist_ok=True)

    for i,(t1_path, t2_path, t1ce_path, flair_path, gt_path) in enumerate(zip(t1_list,t2_list,t1ce_list,flair_list,gt_list)):

        print('preprocessing the',i+1,'th subject')

        t1_img = nib.load(t1_path)  # (240,140,155)
        t2_img = nib.load(t2_path)
        flair_img = nib.load(flair_path)
        t1ce_img = nib.load(t1ce_path)
        gt_img = nib.load(gt_path)

        #to numpy
        t1_data = t1_img.get_fdata()
        t2_data = t2_img.get_fdata()
        flair_data = flair_img.get_fdata()
        t1ce_data = t1ce_img.get_fdata()
        gt_data = gt_img.get_fdata()
        gt_data = gt_data.astype(np.uint8)
        gt_data[gt_data == 4] = 3 #label 3 is missing in BraTS 2020

        t1_data = normalize(t1_data) # normalize to [0,1]
        t2_data = normalize(t2_data)
        t1ce_data = normalize(t1ce_data)
        flair_data = normalize(flair_data)

        tensor = np.stack([t1_data, t2_data, t1ce_data, flair_data, gt_data])  # (4, 240, 240, 155)
        
        if i < train_len:
            for j in range(60):
                Tensor = tensor[:, 10:210, 25:225, 50 + j]
                np.save(train_path + str(60 * i + j + 1) + '.npy', Tensor)
        else:
            for j in range(60):
                Tensor = tensor[:, 10:210, 25:225, 50 + j]
                np.save(test_path + str(60 * (i - train_len) + j + 1) + '.npy', Tensor)
