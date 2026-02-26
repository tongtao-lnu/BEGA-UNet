"""
文件名: models/unet.py
功能: 标准U-Net模型（基线对比用）
"""

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """双卷积块"""

    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class UNet(nn.Module):
    """标准U-Net"""

    def __init__(self, in_channels=1, num_classes=1, base_filters=64):
        super(UNet, self).__init__()

        f = [base_filters * (2 ** i) for i in range(5)]

        # 编码器
        self.enc1 = DoubleConv(in_channels, f[0])
        self.enc2 = DoubleConv(f[0], f[1])
        self.enc3 = DoubleConv(f[1], f[2])
        self.enc4 = DoubleConv(f[2], f[3])

        self.pool = nn.MaxPool2d(2)

        # 瓶颈
        self.bottleneck = DoubleConv(f[3], f[4])

        # 解码器
        self.up4 = nn.ConvTranspose2d(f[4], f[3], kernel_size=2, stride=2)
        self.dec4 = DoubleConv(f[4], f[3])

        self.up3 = nn.ConvTranspose2d(f[3], f[2], kernel_size=2, stride=2)
        self.dec3 = DoubleConv(f[3], f[2])

        self.up2 = nn.ConvTranspose2d(f[2], f[1], kernel_size=2, stride=2)
        self.dec2 = DoubleConv(f[2], f[1])

        self.up1 = nn.ConvTranspose2d(f[1], f[0], kernel_size=2, stride=2)
        self.dec1 = DoubleConv(f[1], f[0])

        # 输出
        self.out = nn.Conv2d(f[0], num_classes, kernel_size=1)

    def forward(self, x):
        # 编码
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        # 瓶颈
        b = self.bottleneck(self.pool(e4))

        # 解码
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return self.out(d1)