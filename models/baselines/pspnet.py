"""
文件名: models/baselines/pspnet.py
功能: PSPNet (Zhao et al., 2017)
论文: Pyramid Scene Parsing Network
会议: CVPR 2017
引用: 7000+
特点: 金字塔池化模块(PPM)捕获多尺度全局上下文信息
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNReLU(nn.Module):
    """卷积 + BN + ReLU"""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                      stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class PyramidPoolingModule(nn.Module):
    """
    金字塔池化模块 (Pyramid Pooling Module, PPM)

    核心思想：使用不同尺度的池化操作捕获全局上下文
    池化尺度: 1x1, 2x2, 3x3, 6x6
    """

    def __init__(self, in_channels, pool_sizes=(1, 2, 3, 6)):
        super(PyramidPoolingModule, self).__init__()

        out_channels = in_channels // len(pool_sizes)

        self.stages = nn.ModuleList()
        for pool_size in pool_sizes:
            self.stages.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(output_size=pool_size),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ))

        # 融合卷积
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels + out_channels * len(pool_sizes),
                      in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)
        )

    def forward(self, x):
        h, w = x.size()[2:]

        # 各尺度池化 + 上采样
        pyramids = [x]
        for stage in self.stages:
            out = stage(x)
            out = F.interpolate(out, size=(h, w), mode='bilinear', align_corners=True)
            pyramids.append(out)

        # 拼接并融合
        out = torch.cat(pyramids, dim=1)
        out = self.bottleneck(out)

        return out


class ResidualBlock(nn.Module):
    """残差块 (简化版)"""

    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

        # 残差连接
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        out = self.relu(out)

        return out


class PSPNet(nn.Module):
    """
    PSPNet: Pyramid Scene Parsing Network

    架构：
    1. 编码器：类ResNet骨干网络提取特征
    2. PPM：金字塔池化模块捕获全局上下文
    3. 解码器：上采样恢复分辨率

    参数:
        in_channels: 输入通道数 (RGB=3)
        num_classes: 输出类别数
        base_filters: 基础滤波器数量
    """

    def __init__(self, in_channels=3, num_classes=1, base_filters=64):
        super(PSPNet, self).__init__()

        f = base_filters  # 64

        # 初始卷积
        self.stem = nn.Sequential(
            ConvBNReLU(in_channels, f, kernel_size=3, stride=2, padding=1),
            ConvBNReLU(f, f, kernel_size=3, padding=1),
            ConvBNReLU(f, f * 2, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # 编码器 (ResNet风格)
        self.layer1 = self._make_layer(f * 2, f * 2, num_blocks=2, stride=1)
        self.layer2 = self._make_layer(f * 2, f * 4, num_blocks=2, stride=2)
        self.layer3 = self._make_layer(f * 4, f * 8, num_blocks=2, stride=2)
        self.layer4 = self._make_layer(f * 8, f * 8, num_blocks=2, stride=1)

        # 金字塔池化模块
        self.ppm = PyramidPoolingModule(f * 8, pool_sizes=(1, 2, 3, 6))

        # 解码器
        self.decoder = nn.Sequential(
            ConvBNReLU(f * 8, f * 4, kernel_size=3, padding=1),
            nn.Dropout2d(0.1),
            ConvBNReLU(f * 4, f * 2, kernel_size=3, padding=1),
            nn.Dropout2d(0.1),
        )

        # 输出层
        self.out_conv = nn.Conv2d(f * 2, num_classes, kernel_size=1)

        # 辅助分支 (训练时使用)
        self.aux_branch = nn.Sequential(
            ConvBNReLU(f * 8, f * 4, kernel_size=3, padding=1),
            nn.Dropout2d(0.1),
            nn.Conv2d(f * 4, num_classes, kernel_size=1)
        )

        self._init_weights()

    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = [ResidualBlock(in_channels, out_channels, stride)]
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        input_size = x.size()[2:]

        # 编码器
        x = self.stem(x)  # /4
        x = self.layer1(x)  # /4
        x = self.layer2(x)  # /8
        x = self.layer3(x)  # /16
        x = self.layer4(x)  # /16

        # 金字塔池化
        x = self.ppm(x)

        # 解码器
        x = self.decoder(x)

        # 上采样到原始尺寸
        x = F.interpolate(x, size=input_size, mode='bilinear', align_corners=True)

        # 输出
        out = self.out_conv(x)

        return out


# 测试代码
if __name__ == "__main__":
    print("=" * 60)
    print("PSPNet 模型测试")
    print("=" * 60)

    model = PSPNet(in_channels=3, num_classes=1, base_filters=64)

    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n参数量: {total_params / 1e6:.2f}M")

    # 测试前向传播
    x = torch.randn(2, 3, 352, 352)
    y = model(x)

    print(f"输入尺寸: {x.shape}")
    print(f"输出尺寸: {y.shape}")

    assert y.shape == (2, 1, 352, 352), "输出尺寸错误！"
    print("\n✅ PSPNet 测试通过！")
    print("=" * 60)