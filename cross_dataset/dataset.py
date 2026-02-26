"""
文件名: cross_dataset/dataset.py
功能: 跨数据集实验专用数据加载器（RGB 3通道 + 强增强）
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A


class CrossDataset(Dataset):
    """跨数据集实验数据集"""

    def __init__(self, data_dir, augment=False, strong_augment=False):
        self.img_dir = os.path.join(data_dir, "images")
        self.mask_dir = os.path.join(data_dir, "masks")

        self.files = sorted([f for f in os.listdir(self.img_dir) if f.endswith('.npy')])

        if strong_augment:
            self.transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.15, scale_limit=0.25, rotate_limit=45, p=0.7, border_mode=0),
                A.ElasticTransform(alpha=120, sigma=6, p=0.3),
                A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.15, p=0.8),
                A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
                A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
                A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=0.5),
                A.GaussNoise(var_limit=(10, 80), p=0.4),
                A.GaussianBlur(blur_limit=(3, 7), p=0.3),
                A.CLAHE(clip_limit=4.0, p=0.3),
                A.RandomGamma(gamma_limit=(80, 120), p=0.3),
            ])
        elif augment:
            self.transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=30, p=0.5, border_mode=0),
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
                A.GaussNoise(var_limit=(10, 50), p=0.3),
            ])
        else:
            self.transform = None

        print(f"[Dataset] 加载 {len(self.files)} 张图像 from {data_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = np.load(os.path.join(self.img_dir, self.files[idx]))
        mask = np.load(os.path.join(self.mask_dir, self.files[idx]))

        if len(img.shape) == 2:
            img = np.stack([img, img, img], axis=-1)
        if len(mask.shape) == 3:
            mask = mask[:, :, 0]

        if self.transform is not None:
            img_uint8 = (img * 255).clip(0, 255).astype(np.uint8)
            mask_uint8 = (mask * 255).clip(0, 255).astype(np.uint8)
            transformed = self.transform(image=img_uint8, mask=mask_uint8)
            img = transformed['image'].astype(np.float32) / 255.0
            mask = (transformed['mask'] > 127).astype(np.float32)

        img_tensor = torch.from_numpy(img.transpose(2, 0, 1)).float()
        mask_tensor = torch.from_numpy(mask[np.newaxis, :, :]).float()

        return img_tensor, mask_tensor, self.files[idx]


def get_cross_dataloaders(train_dir, test_dir, batch_size=8, strong_augment=True):
    """获取跨数据集数据加载器"""
    train_dataset = CrossDataset(train_dir, augment=True, strong_augment=strong_augment)
    test_dataset = CrossDataset(test_dir, augment=False, strong_augment=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False,
                             num_workers=0, pin_memory=True)

    return train_loader, test_loader