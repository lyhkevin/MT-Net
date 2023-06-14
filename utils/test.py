import argparse
from utils.dataloader import *
from math import log10, sqrt
from skimage.metrics import structural_similarity as ssim

def psnr(res,gt):
    mse = np.mean((res - gt) ** 2)
    if(mse == 0):
        return 100
    max_pixel = 1
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr

def mse(res,gt):
    mae = np.sum(np.abs(res - gt)) * 1.0 / (gt.shape[0] * gt.shape[1])
    return mae

def nmse(res,gt):
    Norm = np.linalg.norm((gt * gt),ord=2)
    if np.all(Norm == 0):
        return 0 
    else:
        nmse = np.linalg.norm(((res - gt) * (res - gt)),ord=2) / Norm
    return nmse
