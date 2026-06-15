"""
通用内存映射数据集类
从 .npy 文件直接加载图像（和标签），在 __getitem__ 中动态应用增强。
适用于分类任务（硬标签或软标签）。
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image


class PreloadedDataset(Dataset):
    """
    基于 numpy 内存映射的高效数据集
    Args:
        images_path (str): 图像 .npy 文件路径，形状 (N, H, W)，dtype=uint8
        labels_path (str, optional): 标签 .npy 文件路径，形状 (N,) dtype=int64 或 float32（软标签）
        transform (callable, optional): 增强/预处理函数（通常为 albumentations.Compose）
        soft_labels (bool): 若为 True，labels 应为形状 (N, num_classes) 的浮点数组
    """
    def __init__(self, images_path, labels_path=None, transform=None, soft_labels=False):
        self.images = np.load(images_path)
        self.labels = None
        self.soft_labels = soft_labels

        if labels_path is not None:
            self.labels = np.load(labels_path)
            if not soft_labels:
                # 确保硬标签为 int64
                self.labels = self.labels.astype(np.int64)

        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # 读取图像（uint8, HxW）
        img = self.images[idx]
        # 转为 PIL 灰度图，以兼容 albumentations
        img = Image.fromarray(img, mode='L')

        if self.transform:
            img = self.transform(image=np.array(img))['image']
        else:
            # 默认转为张量并归一化（如果无 transform）
            img = torch.from_numpy(np.array(img, dtype=np.float32) / 255.0).unsqueeze(0)

        if self.labels is not None:
            label = self.labels[idx]
            if self.soft_labels:
                label = torch.as_tensor(label, dtype=torch.float32)
            else:
                label = torch.as_tensor(label, dtype=torch.long)
            return img, label
        return img