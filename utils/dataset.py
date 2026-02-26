"""
文件名: utils/dataset.py
功能: 息肉分割数据集加载与增强
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A


class MedicalDataset(Dataset):
    """医学图像分割数据集"""

    def __init__(self, data_dir, split='train', augment=True):
        self.img_dir = os.path.join(data_dir, split, 'images')
        self.mask_dir = os.path.join(data_dir, split, 'masks')
        self.augment = augment and (split == 'train')

        # 获取文件列表
        if os.path.exists(self.img_dir):
            self.files = sorted([f for f in os.listdir(self.img_dir) if f.endswith('.npy')])
        else:
            self.files = []
            print(f"警告: 目录不存在 {self.img_dir}")

        # 数据增强（仅训练集）
        if self.augment:
            self.transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.1,
                    scale_limit=0.2,
                    rotate_limit=30,
                    p=0.5,
                    border_mode=0
                ),
                A.OneOf([
                    A.GaussNoise(var_limit=(10, 50), p=1),
                    A.GaussianBlur(blur_limit=(3, 5), p=1),
                ], p=0.3),
                A.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                    hue=0.1,
                    p=0.4
                ),
                A.RandomBrightnessContrast(
                    brightness_limit=0.2,
                    contrast_limit=0.2,
                    p=0.3
                ),
            ])
        else:
            self.transform = None

        print(f"[{split}] 加载 {len(self.files)} 张图像")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # 加载数据
        img_path = os.path.join(self.img_dir, self.files[idx])
        mask_path = os.path.join(self.mask_dir, self.files[idx])

        img = np.load(img_path).astype(np.float32)
        mask = np.load(mask_path).astype(np.float32)

        # 确保图像是3通道
        if len(img.shape) == 2:
            img = np.stack([img, img, img], axis=-1)

        # 确保mask是2D
        if len(mask.shape) == 3:
            mask = mask[:, :, 0]

        # 数据增强
        if self.transform is not None:
            # albumentations需要uint8格式进行某些变换
            img_uint8 = (img * 255).clip(0, 255).astype(np.uint8)
            mask_uint8 = (mask * 255).clip(0, 255).astype(np.uint8)

            transformed = self.transform(image=img_uint8, mask=mask_uint8)

            img = transformed['image'].astype(np.float32) / 255.0
            mask = transformed['mask'].astype(np.float32) / 255.0

            # 确保mask二值化
            mask = (mask > 0.5).astype(np.float32)

        # 转换为PyTorch张量
        # 图像: (H, W, 3) -> (3, H, W)
        img_tensor = torch.from_numpy(img.transpose(2, 0, 1)).float()

        # 掩码: (H, W) -> (1, H, W)
        mask_tensor = torch.from_numpy(mask[np.newaxis, :, :]).float()

        return img_tensor, mask_tensor


def get_dataloaders(data_dir, batch_size=8, num_workers=0):
    """创建数据加载器"""

    train_dataset = MedicalDataset(data_dir, split='train', augment=True)
    val_dataset = MedicalDataset(data_dir, split='val', augment=False)
    test_dataset = MedicalDataset(data_dir, split='test', augment=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader


# 测试代码
if __name__ == "__main__":
    # 测试数据集加载
    train_loader, val_loader, test_loader = get_dataloaders(
        "./processed_data", batch_size=4
    )

    # 获取一个batch
    for imgs, masks in train_loader:
        print(f"图像shape: {imgs.shape}")  # 应该是 (4, 3, 352, 352)
        print(f"掩码shape: {masks.shape}")  # 应该是 (4, 1, 352, 352)
        print(f"图像范围: [{imgs.min():.3f}, {imgs.max():.3f}]")
        print(f"掩码范围: [{masks.min():.3f}, {masks.max():.3f}]")
        break