# 导入库
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch import optim
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")
import os.path as osp
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
 
 
torch.manual_seed(17)
# 自定义数据集CamVidDataset
class CamVidDataset(torch.utils.data.Dataset):
    """CamVid Dataset. Read images, apply augmentation and preprocessing transformations.

    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_values (list): values of classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing
            (e.g. noralization, shape manipulation, etc.)
    """

    def __init__(self, images_dir, masks_dir, leaf_dir):
        self.transform = A.Compose([
            A.Resize(224, 224),
            A.HorizontalFlip(),
            A.VerticalFlip(),
            A.Normalize(),
            ToTensorV2(),
        ])
        self.image_ids = [os.path.splitext(file)[0] for file in os.listdir(images_dir)]
        valid_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.PNG'}  # 设置有效的扩展名

        self.images_fps = [
            os.path.join(images_dir, image_id + ext)
            for image_id in self.image_ids
            for ext in valid_extensions
            if os.path.exists(os.path.join(images_dir, image_id + ext))  # 检查文件是否存在
        ]
        self.masks_fps = [os.path.join(masks_dir, image_id + '.png') for image_id in self.image_ids]
        self.leaf_fps = [os.path.join(leaf_dir, image_id + '.png') for image_id in self.image_ids]


    def __getitem__(self, i):
        # read data
        image = np.array(Image.open(self.images_fps[i]).convert('RGB'))

        mask = np.array(Image.open(self.masks_fps[i]).convert('L'))
        # print(mask)
        # 处理标签，将非0值替换为1
        mask = np.where(mask == 38, 1, mask).astype(np.uint8)

        leaf = np.array(Image.open(self.leaf_fps[i]).convert('L'))
        # print(mask)
        # 处理标签，将非0值替换为1
        leaf = np.where(leaf == 38, 1, leaf).astype(np.uint8)

        # 应用变换
        transformed = self.transform(image=image, mask=mask)
        transformed1 = self.transform(image=image, mask=leaf)

        # 返回图像和标签
        # print(transformed['image'].shape, transformed['mask'].squeeze().shape, transformed1['mask'].squeeze().shape)
        return transformed['image'], transformed['mask'].squeeze(), transformed1['mask'].squeeze()

    def __len__(self):
        return len(self.image_ids)
