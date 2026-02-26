"""
文件名: models/baselines/resunet.py
功能: Residual U-Net (Zhang et al., 2018)
论文: Road Extraction by Deep Residual U-Net
"""

import torch
import torch.nn as nn


class ResidualConvBlock(nn.Module):
    """残差卷积块"""

    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualConvBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = self.shortcut(x)

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        out += residual
        out = self.relu(out)

        return out


class ResUNet(nn.Module):
    """Residual U-Net"""

    def __init__(self, in_channels=1, num_classes=1, base_filters=64):
        super(ResUNet, self).__init__()

        f = [base_filters * (2 ** i) for i in range(5)]

        # 编码器
        self.enc1 = ResidualConvBlock(in_channels, f[0])
        self.enc2 = ResidualConvBlock(f[0], f[1], stride=2)
        self.enc3 = ResidualConvBlock(f[1], f[2], stride=2)
        self.enc4 = ResidualConvBlock(f[2], f[3], stride=2)

        # 瓶颈
        self.bottleneck = ResidualConvBlock(f[3], f[4], stride=2)

        # 解码器
        self.up4 = nn.ConvTranspose2d(f[4], f[3], kernel_size=2, stride=2)
        self.dec4 = ResidualConvBlock(f[4], f[3])

        self.up3 = nn.ConvTranspose2d(f[3], f[2], kernel_size=2, stride=2)
        self.dec3 = ResidualConvBlock(f[3], f[2])

        self.up2 = nn.ConvTranspose2d(f[2], f[1], kernel_size=2, stride=2)
        self.dec2 = ResidualConvBlock(f[2], f[1])

        self.up1 = nn.ConvTranspose2d(f[1], f[0], kernel_size=2, stride=2)
        self.dec1 = ResidualConvBlock(f[1], f[0])

        # 输出
        self.out = nn.Conv2d(f[0], num_classes, kernel_size=1)

    def forward(self, x):
        # 编码
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        # 瓶颈
        b = self.bottleneck(e4)

        # 解码
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return self.out(d1)


if __name__ == "__main__":
    model = ResUNet(in_channels=1, num_classes=1)
    x = torch.randn(2, 1, 512, 512)
    y = model(x)
    print(f"Input: {x.shape}, Output: {y.shape}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")