"""
文件名: models/baselines/attention_unet.py
功能: Attention U-Net (Oktay et al., 2018)
论文: Attention U-Net: Learning Where to Look for the Pancreas
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """双卷积块"""

    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class AttentionGate(nn.Module):
    """注意力门机制"""

    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate, self).__init__()

        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)

        # 确保尺寸匹配
        if g1.shape[2:] != x1.shape[2:]:
            g1 = F.interpolate(g1, size=x1.shape[2:], mode='bilinear', align_corners=True)

        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi


class AttentionUNet(nn.Module):
    """Attention U-Net"""

    def __init__(self, in_channels=1, num_classes=1, base_filters=64):
        super(AttentionUNet, self).__init__()

        f = [base_filters * (2 ** i) for i in range(5)]
        # f = [64, 128, 256, 512, 1024]

        # 编码器
        self.enc1 = ConvBlock(in_channels, f[0])
        self.enc2 = ConvBlock(f[0], f[1])
        self.enc3 = ConvBlock(f[1], f[2])
        self.enc4 = ConvBlock(f[2], f[3])

        self.pool = nn.MaxPool2d(2)

        # 瓶颈
        self.bottleneck = ConvBlock(f[3], f[4])

        # 上采样
        self.up4 = nn.ConvTranspose2d(f[4], f[3], kernel_size=2, stride=2)
        self.up3 = nn.ConvTranspose2d(f[3], f[2], kernel_size=2, stride=2)
        self.up2 = nn.ConvTranspose2d(f[2], f[1], kernel_size=2, stride=2)
        self.up1 = nn.ConvTranspose2d(f[1], f[0], kernel_size=2, stride=2)

        # 注意力门（修正：F_g是上采样后的通道数）
        self.att4 = AttentionGate(F_g=f[3], F_l=f[3], F_int=f[2])
        self.att3 = AttentionGate(F_g=f[2], F_l=f[2], F_int=f[1])
        self.att2 = AttentionGate(F_g=f[1], F_l=f[1], F_int=f[0])
        self.att1 = AttentionGate(F_g=f[0], F_l=f[0], F_int=f[0] // 2)

        # 解码器
        self.dec4 = ConvBlock(f[4], f[3])
        self.dec3 = ConvBlock(f[3], f[2])
        self.dec2 = ConvBlock(f[2], f[1])
        self.dec1 = ConvBlock(f[1], f[0])

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

        # 解码 + 注意力
        d4 = self.up4(b)
        e4_att = self.att4(g=d4, x=e4)
        d4 = self.dec4(torch.cat([d4, e4_att], dim=1))

        d3 = self.up3(d4)
        e3_att = self.att3(g=d3, x=e3)
        d3 = self.dec3(torch.cat([d3, e3_att], dim=1))

        d2 = self.up2(d3)
        e2_att = self.att2(g=d2, x=e2)
        d2 = self.dec2(torch.cat([d2, e2_att], dim=1))

        d1 = self.up1(d2)
        e1_att = self.att1(g=d1, x=e1)
        d1 = self.dec1(torch.cat([d1, e1_att], dim=1))

        return self.out(d1)


if __name__ == "__main__":
    model = AttentionUNet(in_channels=1, num_classes=1)
    x = torch.randn(2, 1, 512, 512)
    y = model(x)
    print(f"Input: {x.shape}, Output: {y.shape}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")