"""
文件名: models/baselines/sanet.py
功能: SANet (Wei et al., 2021)
论文: Shallow Attention Network for Polyp Segmentation
会议: MICCAI 2021
引用: 400+
特点: 浅层注意力模块(SA) + 概率校正模块(PC)，轻量级高效设计
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNReLU(nn.Module):
    """卷积 + BN + ReLU"""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                      stride=stride, padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class ShallowAttention(nn.Module):
    """
    浅层注意力模块 (Shallow Attention Module)

    核心思想：在浅层特征上应用注意力，保留更多细节信息
    """

    def __init__(self, in_channels, out_channels):
        super(ShallowAttention, self).__init__()

        # 空间注意力
        self.spatial_att = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, 1),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, 1, 1),
            nn.Sigmoid()
        )

        # 通道注意力
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, in_channels, 1),
            nn.Sigmoid()
        )

        # 特征变换
        self.conv = ConvBNReLU(in_channels, out_channels, 3, padding=1)

    def forward(self, x):
        # 空间注意力
        sa = self.spatial_att(x)
        x = x * sa

        # 通道注意力
        ca = self.channel_att(x)
        x = x * ca

        # 特征变换
        x = self.conv(x)
        return x


class ProbabilityCorrection(nn.Module):
    """
    概率校正模块 (Probability Correction Module)

    通过高层语义信息校正低层预测
    """

    def __init__(self, in_channels, out_channels):
        super(ProbabilityCorrection, self).__init__()

        self.conv1 = ConvBNReLU(in_channels, out_channels, 3, padding=1)
        self.conv2 = ConvBNReLU(out_channels, out_channels, 3, padding=1)
        self.conv3 = nn.Conv2d(out_channels, 1, 1)

    def forward(self, x, prob_map=None):
        if prob_map is not None:
            prob_map = F.interpolate(prob_map, size=x.size()[2:], mode='bilinear', align_corners=True)
            x = x * torch.sigmoid(prob_map) + x

        x = self.conv1(x)
        x = self.conv2(x)
        out = self.conv3(x)
        return out, x


class Encoder(nn.Module):
    """简化编码器"""

    def __init__(self, in_channels=3):
        super(Encoder, self).__init__()

        self.stage1 = nn.Sequential(
            ConvBNReLU(in_channels, 64, 3, stride=2, padding=1),
            ConvBNReLU(64, 64, 3, padding=1),
        )

        self.stage2 = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBNReLU(64, 128, 3, padding=1),
            ConvBNReLU(128, 128, 3, padding=1),
        )

        self.stage3 = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBNReLU(128, 256, 3, padding=1),
            ConvBNReLU(256, 256, 3, padding=1),
            ConvBNReLU(256, 256, 3, padding=1),
        )

        self.stage4 = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBNReLU(256, 512, 3, padding=1),
            ConvBNReLU(512, 512, 3, padding=1),
            ConvBNReLU(512, 512, 3, padding=1),
        )

        self.stage5 = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBNReLU(512, 512, 3, padding=1),
            ConvBNReLU(512, 512, 3, padding=1),
            ConvBNReLU(512, 512, 3, padding=1),
        )

    def forward(self, x):
        x1 = self.stage1(x)  # 1/2
        x2 = self.stage2(x1)  # 1/4
        x3 = self.stage3(x2)  # 1/8
        x4 = self.stage4(x3)  # 1/16
        x5 = self.stage5(x4)  # 1/32
        return x1, x2, x3, x4, x5


class SANet(nn.Module):
    """
    SANet: Shallow Attention Network

    架构：
    1. 编码器提取多尺度特征
    2. 浅层注意力模块增强细节
    3. 概率校正模块逐步细化预测

    参数:
        in_channels: 输入通道数 (RGB=3)
        num_classes: 输出类别数
        base_filters: 基础滤波器数量
    """

    def __init__(self, in_channels=3, num_classes=1, base_filters=64):
        super(SANet, self).__init__()

        # 编码器
        self.encoder = Encoder(in_channels)

        # 浅层注意力模块
        self.sa5 = ShallowAttention(512, 256)
        self.sa4 = ShallowAttention(512, 256)
        self.sa3 = ShallowAttention(256, 128)
        self.sa2 = ShallowAttention(128, 64)

        # 概率校正模块
        self.pc5 = ProbabilityCorrection(256, 128)
        self.pc4 = ProbabilityCorrection(256, 128)
        self.pc3 = ProbabilityCorrection(128, 64)
        self.pc2 = ProbabilityCorrection(64, 32)

        # 上采样
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x):
        input_size = x.size()[2:]

        # 编码
        x1, x2, x3, x4, x5 = self.encoder(x)

        # 解码 + 浅层注意力 + 概率校正
        # Stage 5
        f5 = self.sa5(x5)
        out5, f5 = self.pc5(f5)

        # Stage 4
        f4 = self.up(f5)
        f4 = self.sa4(x4)
        out4, f4 = self.pc4(f4, out5)

        # Stage 3
        f3 = self.up(f4)
        f3 = self.sa3(x3)
        out3, f3 = self.pc3(f3, out4)

        # Stage 2
        f2 = self.up(f3)
        f2 = self.sa2(x2)
        out2, f2 = self.pc2(f2, out3)

        # 上采样到原始尺寸
        out = F.interpolate(out2, size=input_size, mode='bilinear', align_corners=True)

        return out


# 测试代码
if __name__ == "__main__":
    print("=" * 60)
    print("SANet 模型测试")
    print("=" * 60)

    model = SANet(in_channels=3, num_classes=1, base_filters=64)

    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n参数量: {total_params / 1e6:.2f}M")

    # 测试前向传播
    x = torch.randn(2, 3, 352, 352)
    y = model(x)

    print(f"输入尺寸: {x.shape}")
    print(f"输出尺寸: {y.shape}")

    assert y.shape == (2, 1, 352, 352), "输出尺寸错误！"
    print("\n✅ SANet 测试通过！")
    print("=" * 60)