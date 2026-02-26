"""
文件名: models/baselines/fapnet.py
功能: FAPNet (Feature Aggregation and Propagation Network)
论文: FAPNet: Feature Aggregation and Propagation Network for Polyp Segmentation
来源: arXiv 2025.01
特点:
    - 特征聚合模块(FAM)：高效融合多尺度特征
    - 特征传播模块(FPM)：逐层传递语义信息
    - 边界感知损失：隐式边界学习
    - 轻量级设计：参数少，推理快

优势: 纯CNN架构，训练稳定，速度快
劣势: 没有显式边缘检测模块（EGA-UNet的优势所在）
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


class ChannelAttention(nn.Module):
    """通道注意力模块"""

    def __init__(self, channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return x * self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    """空间注意力模块"""

    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        concat = torch.cat([avg_out, max_out], dim=1)
        return x * self.sigmoid(self.conv(concat))


class FeatureAggregationModule(nn.Module):
    """
    特征聚合模块 (Feature Aggregation Module, FAM)

    核心思想：使用多尺度卷积和注意力机制聚合不同层次的特征
    """

    def __init__(self, in_channels, out_channels):
        super(FeatureAggregationModule, self).__init__()

        # 多尺度卷积分支
        self.branch1 = ConvBNReLU(in_channels, out_channels // 4, kernel_size=1, padding=0)
        self.branch2 = ConvBNReLU(in_channels, out_channels // 4, kernel_size=3, padding=1)
        self.branch3 = ConvBNReLU(in_channels, out_channels // 4, kernel_size=3, padding=2, dilation=2)
        self.branch4 = ConvBNReLU(in_channels, out_channels // 4, kernel_size=3, padding=3, dilation=3)

        # 融合卷积
        self.fuse = ConvBNReLU(out_channels, out_channels, kernel_size=3, padding=1)

        # 注意力
        self.ca = ChannelAttention(out_channels)
        self.sa = SpatialAttention()

        # 残差连接
        self.residual = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        ) if in_channels != out_channels else nn.Identity()

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # 多尺度特征提取
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)

        # 拼接
        out = torch.cat([b1, b2, b3, b4], dim=1)
        out = self.fuse(out)

        # 注意力增强
        out = self.ca(out)
        out = self.sa(out)

        # 残差
        out = self.relu(out + self.residual(x))

        return out


class FeaturePropagationModule(nn.Module):
    """
    特征传播模块 (Feature Propagation Module, FPM)

    核心思想：将高层语义信息逐层传播到低层，增强特征表示
    """

    def __init__(self, high_channels, low_channels, out_channels):
        super(FeaturePropagationModule, self).__init__()

        # 高层特征处理
        self.high_conv = nn.Sequential(
            ConvBNReLU(high_channels, out_channels, kernel_size=3, padding=1),
            ConvBNReLU(out_channels, out_channels, kernel_size=3, padding=1),
        )

        # 低层特征处理
        self.low_conv = nn.Sequential(
            ConvBNReLU(low_channels, out_channels, kernel_size=3, padding=1),
            ConvBNReLU(out_channels, out_channels, kernel_size=3, padding=1),
        )

        # 融合门控
        self.gate = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.Sigmoid()
        )

        # 输出卷积
        self.out_conv = ConvBNReLU(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, high_feat, low_feat):
        # 上采样高层特征
        high_feat = F.interpolate(high_feat, size=low_feat.shape[2:],
                                  mode='bilinear', align_corners=True)

        # 特征变换
        high_feat = self.high_conv(high_feat)
        low_feat = self.low_conv(low_feat)

        # 门控融合
        gate = self.gate(torch.cat([high_feat, low_feat], dim=1))
        out = high_feat * gate + low_feat * (1 - gate)

        # 输出
        out = self.out_conv(out)

        return out


class Encoder(nn.Module):
    """轻量级编码器"""

    def __init__(self, in_channels=3):
        super(Encoder, self).__init__()

        # 初始层
        self.stem = nn.Sequential(
            ConvBNReLU(in_channels, 32, kernel_size=3, stride=2, padding=1),
            ConvBNReLU(32, 32, kernel_size=3, padding=1),
            ConvBNReLU(32, 64, kernel_size=3, padding=1),
            nn.MaxPool2d(2)
        )

        # 编码器阶段
        self.stage1 = nn.Sequential(
            ConvBNReLU(64, 64, kernel_size=3, padding=1),
            ConvBNReLU(64, 64, kernel_size=3, padding=1),
        )

        self.stage2 = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBNReLU(64, 128, kernel_size=3, padding=1),
            ConvBNReLU(128, 128, kernel_size=3, padding=1),
        )

        self.stage3 = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBNReLU(128, 256, kernel_size=3, padding=1),
            ConvBNReLU(256, 256, kernel_size=3, padding=1),
        )

        self.stage4 = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBNReLU(256, 512, kernel_size=3, padding=1),
            ConvBNReLU(512, 512, kernel_size=3, padding=1),
        )

    def forward(self, x):
        x = self.stem(x)  # 1/4
        x1 = self.stage1(x)  # 64, 1/4
        x2 = self.stage2(x1)  # 128, 1/8
        x3 = self.stage3(x2)  # 256, 1/16
        x4 = self.stage4(x3)  # 512, 1/32
        return x1, x2, x3, x4


class FAPNet(nn.Module):
    """
    FAPNet: Feature Aggregation and Propagation Network

    架构：
    1. 轻量级编码器提取多尺度特征
    2. FAM模块在各层进行特征聚合
    3. FPM模块逐层传播语义信息
    4. 最终预测头输出分割结果

    参数:
        in_channels: 输入通道数 (RGB=3)
        num_classes: 输出类别数
        base_filters: 基础滤波器数量
    """

    def __init__(self, in_channels=3, num_classes=1, base_filters=64):
        super(FAPNet, self).__init__()

        # 编码器
        self.encoder = Encoder(in_channels)

        # 特征聚合模块
        self.fam4 = FeatureAggregationModule(512, 256)
        self.fam3 = FeatureAggregationModule(256, 128)
        self.fam2 = FeatureAggregationModule(128, 64)
        self.fam1 = FeatureAggregationModule(64, 64)

        # 特征传播模块
        self.fpm4 = FeaturePropagationModule(256, 128, 128)
        self.fpm3 = FeaturePropagationModule(128, 64, 64)
        self.fpm2 = FeaturePropagationModule(64, 64, 64)

        # 输出层
        self.out_conv = nn.Sequential(
            ConvBNReLU(64, 32, kernel_size=3, padding=1),
            nn.Conv2d(32, num_classes, kernel_size=1)
        )

    def forward(self, x):
        input_size = x.size()[2:]

        # 编码
        x1, x2, x3, x4 = self.encoder(x)
        # x1: (B, 64, H/4, W/4)
        # x2: (B, 128, H/8, W/8)
        # x3: (B, 256, H/16, W/16)
        # x4: (B, 512, H/32, W/32)

        # 特征聚合
        f4 = self.fam4(x4)  # (B, 256, H/32, W/32)
        f3 = self.fam3(x3)  # (B, 128, H/16, W/16)
        f2 = self.fam2(x2)  # (B, 64, H/8, W/8)
        f1 = self.fam1(x1)  # (B, 64, H/4, W/4)

        # 特征传播（自顶向下）
        p3 = self.fpm4(f4, f3)  # (B, 128, H/16, W/16)
        p2 = self.fpm3(p3, f2)  # (B, 64, H/8, W/8)
        p1 = self.fpm2(p2, f1)  # (B, 64, H/4, W/4)

        # 输出
        out = self.out_conv(p1)
        out = F.interpolate(out, size=input_size, mode='bilinear', align_corners=True)

        return out


# 测试代码
if __name__ == "__main__":
    print("=" * 60)
    print("FAPNet 模型测试")
    print("=" * 60)

    model = FAPNet(in_channels=3, num_classes=1, base_filters=64)

    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n参数量: {total_params / 1e6:.2f}M")

    # 测试前向传播
    x = torch.randn(2, 3, 352, 352)
    print(f"输入尺寸: {x.shape}")

    # 详细检查各阶段输出
    model.eval()
    with torch.no_grad():
        # 编码器
        x1, x2, x3, x4 = model.encoder(x)
        print(f"编码器输出: x1={x1.shape}, x2={x2.shape}, x3={x3.shape}, x4={x4.shape}")

        # FAM
        f4 = model.fam4(x4)
        f3 = model.fam3(x3)
        f2 = model.fam2(x2)
        f1 = model.fam1(x1)
        print(f"FAM输出: f1={f1.shape}, f2={f2.shape}, f3={f3.shape}, f4={f4.shape}")

        # 完整前向传播
        y = model(x)

    print(f"最终输出尺寸: {y.shape}")

    assert y.shape == (2, 1, 352, 352), "输出尺寸错误！"
    print("\n✅ FAPNet 测试通过！")
    print("=" * 60)