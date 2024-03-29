# MT-Net
We provide Pytorch implementations for our paper [Multi-scale Transformer Network for Cross-Modality MR Image Synthesis](https://ieeexplore.ieee.org/document/10158035) (IEEE TMI) by Yonghao Li, Tao Zhou, Kelei He, Yi Zhou, and Dinggang Shen.
  
## 1. Introduction

We propose MT-Net to leverage to take advantage of both paired and unpaired data in MRI synthesis. 

<p align="center">
    <img src="imgs/framework.png"/> <br />
    <em> 
    Figure 2. An overview of the proposed MT-Net.
    </em>
</p>

**Preview:**

Our proposed methods consist of two main components under two different settings: 

- Edge-MAE (Self-supervised pre-training with image imputation and edge map estimation).

- MT-Net (Cross-modality MR image synthesis)

Note that our pre-trained Edge-MAE can be utilized for various downstream tasks, such as segmentation or classification.

<p align="center">
    <img src="imgs/EdgeMAE.png"/> <br />
    <em> 
    Figure 2. Example of imputed image and estimated edge maps from the BraTS2020 dataset.
    </em>
</p>

## 2. Getting Started

- ### Installation

  Install PyTorch and torchvision from http://pytorch.org and other dependencies. You can install all the dependencies by
  ```bash
  pip install -r requirements.txt
  ```
  
- ### Dataset Preparation

  Download [BraTS2020](https://www.kaggle.com/datasets/awsaf49/brats20-dataset-training-validation?resource=download) dataset from kaggle. The file name should be `./data/archive.zip`. Unzip the file in `./data/`. 

- ### Date Preprocessing

  After preparing all the data, run the `./utils/preprocessing.py` to normalize the data to [0,1] and crop out an image of size 200×200 from the center.

- ### Pre-training

  To pre-train our Edge-MAE, run `pretrain.py`. You may change the default settings in the `./options/pretrain_options.py`. For instance, increase `num_workers` to speed up fine-tuning.  The weights will be saved in `./weight/EdgeMAE/`. You can also use the pre-trained checkpoints of Edge-MAE in the `./weight/EdgeMAE/`. 

- ### Fine-tuning

   To fine-tune our MT-Net, run `Finetune.py`. You may change the default settings in the `./options/finetune_options.py`, especially the `data_rate` option to adjust the amount of paired data for fine-tuning. Besides, Besides, you can increase `num_workers` to speed up fine-tuning. The weights will be saved in `./weight/finetuned/`. Note that for MT-Net, the input size must be 256×256.

- ### Test

  When fine-tuning is completed, the weights of Edge-MAE and MT-Net will be saved in `./weight/finetune/`. You can change the default settings in the `./options/test_options.py`. Then, run `test.py`, and the synthesized image will be saved in `./snapshot/test/`, and can obtain the PSNR, SSIM, and NMSE values.

## 3. Mindspore
  We also provide Mindspore implementations for our paper, which is a new open source deep learning training/inference framework.
- ### Installation
  Install mindspore from https://www.mindspore.cn/ and other dependencies. Unzip `./mindspore/mindcv.zip`, an open-source toolbox for computer vision research.
- ### Pre-training
  After preparing all the data, run `./mindspore/pretrain.py` to pre-train our EdgeMAE. You may change the default settings in `./mindspore/pretrain.py`.
- ### Fine-tuning
  When pre-training is completed, run `./mindspore/finetune.py`. You may change the default settings in `./mindspore/finetune.py`.
- ### Test
  When fine-tuning is completed, run `./mindspore/test.py`. You may change the default settings in `./mindspore/test.py`.

## 4. Citation

```bibtex
@ARTICLE{10158035,
  author={Li, Yonghao and Zhou, Tao and He, Kelei and Zhou, Yi and Shen, Dinggang},
  journal={IEEE Transactions on Medical Imaging}, 
  title={Multi-scale Transformer Network with Edge-aware Pre-training for Cross-Modality MR Image Synthesis}, 
  year={2023},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/TMI.2023.3288001}}
```

## 5. References
- BraTs 2020: [[HERE]](https://www.med.upenn.edu/cbica/brats2020/data.html)

- MAE: [[HERE]](https://github.com/facebookresearch/mae)

- SwinUNet: [[HERE]](https://github.com/HuCaoFighting/Swin-Unet)
