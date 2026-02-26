"""
文件名: models/baselines/uacanet.py
功能: UACANet (Kim et al., 2021)
论文: UACANet: Uncertainty Augmented Context Attention for Polyp Segmentation
会议: ACM Multimedia 2021
引用: 300+
特点: 不确定性增强 + 上下文注意力模块，处理边界模糊问题
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


class UncertaintyModule(nn.Module):
    """
    不确定性模块

    核心思想：预测像素级不确定性，用于加权损失和特征增强
    """

    def __init__(self, in_channels):
        super(UncertaintyModule, self).__init__()
        self.conv1 = ConvBNReLU(in_channels, in_channels // 2, 3, padding=1)
        self.conv2 = nn.Conv2d(in_channels // 2, 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        unc = self.conv1(x)
        unc = self.conv2(unc)
        unc = self.sigmoid(unc)
        return unc


class ContextAttention(nn.Module):
    """
    上下文注意力模块

    结合局部和全局上下文信息
    """

    def __init__(self, in_channels, out_channels):
        super(ContextAttention, self).__init__()

        # 局部上下文
        self.local_att = nn.Sequential(
            ConvBNReLU(in_channels, in_channels // 4, 1, padding=0),
            ConvBNReLU(in_channels // 4, in_channels // 4, 3, padding=1),
            ConvBNReLU(in_channels // 4, in_channels // 4, 3, padding=1),
            nn.Conv2d(in_channels // 4, 1, 1),
            nn.Sigmoid()
        )

        # 全局上下文
        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, in_channels, 1),
            nn.Sigmoid()
        )

        # 特征变换
        self.conv = ConvBNReLU(in_channels, out_channels, 3, padding=1)

    def forward(self, x):
        # 局部注意力
        local_att = self.local_att(x)
        x_local = x * local_att

        # 全局注意力
        global_att = self.global_att(x)
        x_global = x * global_att

        # 融合
        out = x_local + x_global + x
        out = self.conv(out)

        return out


class UACA(nn.Module):
    """
    不确定性增强上下文注意力 (Uncertainty Augmented Context Attention)
    """

    def __init__(self, in_channels, out_channels):
        super(UACA, self).__init__()

        self.uncertainty = UncertaintyModule(in_channels)
        self.context_att = ContextAttention(in_channels, out_channels)
        self.conv_out = nn.Conv2d(out_channels, 1, 1)

    def forward(self, x, prev_map=None):
        # 不确定性预测
        unc = self.uncertainty(x)

        # 使用不确定性增强特征
        if prev_map is not None:
            prev_map = F.interpolate(prev_map, size=x.size()[2:], mode='bilinear', align_corners=True)
            # 高不确定性区域获得更多关注
            x = x * (1 + unc * torch.sigmoid(prev_map))
        else:
            x = x * (1 + unc)

        # 上下文注意力
        feat = self.context_att(x)
        out = self.conv_out(feat)

        return out, feat, unc


class ResBlock(nn.Module):
    """残差块"""

    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlock, self).__init__()
        self.conv1 = ConvBNReLU(in_channels, out_channels, 3, stride=stride, padding=1)
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.relu = nn.ReLU(inplace=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class Encoder(nn.Module):
    """ResNet风格编码器"""

    def __init__(self, in_channels=3):
        super(Encoder, self).__init__()

        self.stem = nn.Sequential(
            ConvBNReLU(in_channels, 64, 7, stride=2, padding=3),
            nn.MaxPool2d(3, stride=2, padding=1)
        )

        self.layer1 = self._make_layer(64, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)

    def _make_layer(self, in_ch, out_ch, blocks, stride=1):
        layers = [ResBlock(in_ch, out_ch, stride)]
        for _ in range(1, blocks):
            layers.append(ResBlock(out_ch, out_ch))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        return x1, x2, x3, x4


class UACANet(nn.Module):
    """
    UACANet: Uncertainty Augmented Context Attention Network

    架构：
    1. ResNet编码器
    2. 多级UACA模块
    3. 不确定性引导的渐进式解码

    参数:
        in_channels: 输入通道数 (RGB=3)
        num_classes: 输出类别数
        base_filters: 基础滤波器数量
    """

    def __init__(self, in_channels=3, num_classes=1, base_filters=64):
        super(UACANet, self).__init__()

        # 编码器
        self.encoder = Encoder(in_channels)

        # UACA模块
        self.uaca4 = UACA(512, 256)
        self.uaca3 = UACA(256, 128)
        self.uaca2 = UACA(128, 64)
        self.uaca1 = UACA(64, 32)

        # 上采样
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x):
        input_size = x.size()[2:]

        # 编码
        x1, x2, x3, x4 = self.encoder(x)

        # UACA解码
        out4, feat4, unc4 = self.uaca4(x4)

        out3, feat3, unc3 = self.uaca3(x3, out4)

        out2, feat2, unc2 = self.uaca2(x2, out3)

        out1, feat1, unc1 = self.uaca1(x1, out2)

        # 上采样到原始尺寸
        out = F.interpolate(out1, size=input_size, mode='bilinear', align_corners=True)

        return out


# 测试代码
if __name__ == "__main__":
    print("=" * 60)
    print("UACANet 模型测试")
    print("=" * 60)

    model = UACANet(in_channels=3, num_classes=1, base_filters=64)

    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n参数量: {total_params / 1e6:.2f}M")

    # 测试前向传播
    x = torch.randn(2, 3, 352, 352)
    y = model(x)

    print(f"输入尺寸: {x.shape}")
    print(f"输出尺寸: {y.shape}")

    assert y.shape == (2, 1, 352, 352), "输出尺寸错误！"
    print("\n✅ UACANet 测试通过！")
    print("=" * 60)