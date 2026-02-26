"""
文件名: models/baselines/multiresunet.py
功能: MultiResUNet (Ibtehaz & Rahman, 2020)
论文: MultiResUNet: Rethinking the U-Net Architecture for Multiresolution Image Segmentation
期刊: Neural Networks, 2020 (IF ~7.8)
引用: 1800+
特点: 多分辨率残差块(MultiRes Block) + 残差路径(Res Path)，专为医学图像设计
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNReLU(nn.Module):
    """卷积 + BN + ReLU"""

    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                      padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class MultiResBlock(nn.Module):
    """
    多分辨率残差块 (MultiRes Block)

    核心思想：使用不同大小的卷积核（3x3级联模拟5x5和7x7）
    捕获多尺度特征，类似Inception但更高效
    """

    def __init__(self, in_channels, out_channels, alpha=1.67):
        super(MultiResBlock, self).__init__()

        # 计算各分支通道数 (论文公式)
        W = alpha * out_channels

        self.out_channels = int(W * 0.167) + int(W * 0.333) + int(W * 0.5)

        # 3x3卷积分支
        self.conv3x3 = ConvBNReLU(in_channels, int(W * 0.167), kernel_size=3, padding=1)

        # 5x5卷积 (用两个3x3级联)
        self.conv5x5 = ConvBNReLU(int(W * 0.167), int(W * 0.333), kernel_size=3, padding=1)

        # 7x7卷积 (用三个3x3级联)
        self.conv7x7 = ConvBNReLU(int(W * 0.333), int(W * 0.5), kernel_size=3, padding=1)

        # 残差连接的1x1卷积
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, self.out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.out_channels)
        )

        # 最终BN和激活
        self.bn = nn.BatchNorm2d(self.out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # 多尺度卷积
        out3x3 = self.conv3x3(x)
        out5x5 = self.conv5x5(out3x3)
        out7x7 = self.conv7x7(out5x5)

        # 拼接多尺度特征
        out = torch.cat([out3x3, out5x5, out7x7], dim=1)
        out = self.bn(out)

        # 残差连接
        shortcut = self.shortcut(x)
        out = out + shortcut
        out = self.relu(out)

        return out


class ResPath(nn.Module):
    """
    残差路径 (Res Path)

    解决编码器和解码器之间的语义鸿沟
    通过多个残差块逐步调整特征
    """

    def __init__(self, in_channels, out_channels, length):
        super(ResPath, self).__init__()

        self.blocks = nn.ModuleList()
        self.shortcuts = nn.ModuleList()

        for i in range(length):
            if i == 0:
                self.blocks.append(ConvBNReLU(in_channels, out_channels, kernel_size=3, padding=1))
                self.shortcuts.append(nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                    nn.BatchNorm2d(out_channels)
                ))
            else:
                self.blocks.append(ConvBNReLU(out_channels, out_channels, kernel_size=3, padding=1))
                self.shortcuts.append(nn.Sequential(
                    nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
                    nn.BatchNorm2d(out_channels)
                ))

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        for block, shortcut in zip(self.blocks, self.shortcuts):
            out = block(x)
            res = shortcut(x)
            x = self.relu(out + res)
        return x


class MultiResUNet(nn.Module):
    """
    MultiResUNet: 多分辨率U-Net

    主要创新：
    1. MultiRes Block: 替代标准卷积块，捕获多尺度特征
    2. Res Path: 替代简单跳跃连接，减少语义鸿沟

    参数:
        in_channels: 输入通道数 (RGB=3)
        num_classes: 输出类别数
        base_filters: 基础滤波器数量
    """

    def __init__(self, in_channels=3, num_classes=1, base_filters=32):
        super(MultiResUNet, self).__init__()

        # 通道数配置
        f = base_filters  # 32

        # 编码器
        self.mres1 = MultiResBlock(in_channels, f)  # 32 -> ~51
        self.pool1 = nn.MaxPool2d(2)
        self.respath1 = ResPath(self.mres1.out_channels, f, length=4)

        self.mres2 = MultiResBlock(self.mres1.out_channels, f * 2)  # -> ~103
        self.pool2 = nn.MaxPool2d(2)
        self.respath2 = ResPath(self.mres2.out_channels, f * 2, length=3)

        self.mres3 = MultiResBlock(self.mres2.out_channels, f * 4)  # -> ~206
        self.pool3 = nn.MaxPool2d(2)
        self.respath3 = ResPath(self.mres3.out_channels, f * 4, length=2)

        self.mres4 = MultiResBlock(self.mres3.out_channels, f * 8)  # -> ~413
        self.pool4 = nn.MaxPool2d(2)
        self.respath4 = ResPath(self.mres4.out_channels, f * 8, length=1)

        # 瓶颈
        self.mres5 = MultiResBlock(self.mres4.out_channels, f * 16)  # -> ~826

        # 解码器
        self.up4 = nn.ConvTranspose2d(self.mres5.out_channels, f * 8, kernel_size=2, stride=2)
        self.mres6 = MultiResBlock(f * 8 + f * 8, f * 8)

        self.up3 = nn.ConvTranspose2d(self.mres6.out_channels, f * 4, kernel_size=2, stride=2)
        self.mres7 = MultiResBlock(f * 4 + f * 4, f * 4)

        self.up2 = nn.ConvTranspose2d(self.mres7.out_channels, f * 2, kernel_size=2, stride=2)
        self.mres8 = MultiResBlock(f * 2 + f * 2, f * 2)

        self.up1 = nn.ConvTranspose2d(self.mres8.out_channels, f, kernel_size=2, stride=2)
        self.mres9 = MultiResBlock(f + f, f)

        # 输出层
        self.out_conv = nn.Conv2d(self.mres9.out_channels, num_classes, kernel_size=1)

    def forward(self, x):
        # 编码器
        e1 = self.mres1(x)
        p1 = self.pool1(e1)
        s1 = self.respath1(e1)

        e2 = self.mres2(p1)
        p2 = self.pool2(e2)
        s2 = self.respath2(e2)

        e3 = self.mres3(p2)
        p3 = self.pool3(e3)
        s3 = self.respath3(e3)

        e4 = self.mres4(p3)
        p4 = self.pool4(e4)
        s4 = self.respath4(e4)

        # 瓶颈
        b = self.mres5(p4)

        # 解码器
        d4 = self.up4(b)
        d4 = torch.cat([d4, s4], dim=1)
        d4 = self.mres6(d4)

        d3 = self.up3(d4)
        d3 = torch.cat([d3, s3], dim=1)
        d3 = self.mres7(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, s2], dim=1)
        d2 = self.mres8(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, s1], dim=1)
        d1 = self.mres9(d1)

        # 输出
        out = self.out_conv(d1)

        return out


# 测试代码
if __name__ == "__main__":
    print("=" * 60)
    print("MultiResUNet 模型测试")
    print("=" * 60)

    model = MultiResUNet(in_channels=3, num_classes=1, base_filters=32)

    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n参数量: {total_params / 1e6:.2f}M")

    # 测试前向传播
    x = torch.randn(2, 3, 352, 352)
    y = model(x)

    print(f"输入尺寸: {x.shape}")
    print(f"输出尺寸: {y.shape}")

    assert y.shape == (2, 1, 352, 352), "输出尺寸错误！"
    print("\n✅ MultiResUNet 测试通过！")
    print("=" * 60)