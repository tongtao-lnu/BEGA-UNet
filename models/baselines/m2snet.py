"""
文件名: models/baselines/m2snet.py
功能: M2SNet (Zhao et al., 2024)
论文: M2SNet: Multi-scale Subtraction Network for Medical Image Segmentation
期刊: Expert Systems With Applications, 2024 (SCI 1区, IF ~8.5)
引用: 100+ (截至2025)
特点:
    - 多尺度相减模块(MSS)：通过特征相减突出差异
    - 双向融合解码器(BFD)：增强特征融合
    - 轻量级设计：参数量小，推理快
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


class MultiScaleSubtraction(nn.Module):
    """
    多尺度相减模块 (Multi-Scale Subtraction Module)

    核心思想：通过相邻尺度特征的相减操作突出边界和细节差异
    这比简单的加法/拼接更能捕获语义变化
    """

    def __init__(self, in_channels, out_channels):
        super(MultiScaleSubtraction, self).__init__()

        # 多尺度分支
        self.branch1 = ConvBNReLU(in_channels, out_channels, kernel_size=1, padding=0)
        self.branch2 = ConvBNReLU(in_channels, out_channels, kernel_size=3, padding=1)
        self.branch3 = ConvBNReLU(in_channels, out_channels, kernel_size=3, padding=2, dilation=2)
        self.branch4 = ConvBNReLU(in_channels, out_channels, kernel_size=3, padding=4, dilation=4)

        # 相减后的融合
        self.fuse = nn.Sequential(
            ConvBNReLU(out_channels * 3, out_channels, kernel_size=3, padding=1),
            nn.Conv2d(out_channels, out_channels, kernel_size=1)
        )

        # 残差连接
        self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)

        # 多尺度相减：相邻尺度特征差异
        sub1 = b2 - b1  # 细节差异
        sub2 = b3 - b2  # 中等尺度差异
        sub3 = b4 - b3  # 粗尺度差异

        # 拼接差异特征
        out = torch.cat([sub1, sub2, sub3], dim=1)
        out = self.fuse(out)

        # 残差连接
        out = out + self.residual(x)
        out = self.relu(out)

        return out


class BidirectionalFusionDecoder(nn.Module):
    """
    双向融合解码器 (Bidirectional Fusion Decoder)

    同时利用自顶向下和自底向上的信息流
    """

    def __init__(self, high_channels, low_channels, out_channels):
        super(BidirectionalFusionDecoder, self).__init__()

        # 高层特征处理（自顶向下）
        self.high_conv = ConvBNReLU(high_channels, out_channels, kernel_size=3, padding=1)

        # 低层特征处理（自底向上）
        self.low_conv = ConvBNReLU(low_channels, out_channels, kernel_size=3, padding=1)

        # 注意力权重生成
        self.att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels * 2, out_channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // 4, 2, 1),
            nn.Softmax(dim=1)
        )

        # 输出融合
        self.out_conv = ConvBNReLU(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, high_feat, low_feat):
        # 上采样高层特征
        high_feat = F.interpolate(high_feat, size=low_feat.shape[2:],
                                  mode='bilinear', align_corners=True)

        # 特征变换
        high_feat = self.high_conv(high_feat)
        low_feat = self.low_conv(low_feat)

        # 计算注意力权重
        combined = torch.cat([high_feat, low_feat], dim=1)
        weights = self.att(combined)  # (B, 2, 1, 1)

        # 加权融合
        out = weights[:, 0:1] * high_feat + weights[:, 1:2] * low_feat
        out = self.out_conv(out)

        return out


class Encoder(nn.Module):
    """编码器：使用ResNet风格的下采样"""

    def __init__(self, in_channels=3):
        super(Encoder, self).__init__()

        # 初始层
        self.stem = nn.Sequential(
            ConvBNReLU(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.MaxPool2d(3, stride=2, padding=1)
        )

        # 编码器阶段
        self.stage1 = self._make_stage(64, 64, 2)
        self.stage2 = self._make_stage(64, 128, 2, stride=2)
        self.stage3 = self._make_stage(128, 256, 2, stride=2)
        self.stage4 = self._make_stage(256, 512, 2, stride=2)

    def _make_stage(self, in_ch, out_ch, num_blocks, stride=1):
        layers = [ConvBNReLU(in_ch, out_ch, kernel_size=3, stride=stride, padding=1)]
        for _ in range(1, num_blocks):
            layers.append(ConvBNReLU(out_ch, out_ch, kernel_size=3, padding=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x1 = self.stage1(x)  # 64通道, 1/4
        x2 = self.stage2(x1)  # 128通道, 1/8
        x3 = self.stage3(x2)  # 256通道, 1/16
        x4 = self.stage4(x3)  # 512通道, 1/32
        return x1, x2, x3, x4


class M2SNet(nn.Module):
    """
    M2SNet: Multi-scale Subtraction Network

    架构：
    1. 编码器：多阶段特征提取
    2. MSS模块：多尺度相减增强边界
    3. BFD解码器：双向融合恢复细节

    通道流程：
    编码器: 64 -> 128 -> 256 -> 512
    MSS:    64 -> 128 -> 256 -> 512 (保持通道数不变，便于融合)
    BFD:    逐步降低通道数

    参数:
        in_channels: 输入通道数 (RGB=3)
        num_classes: 输出类别数
        base_filters: 基础滤波器数量
    """

    def __init__(self, in_channels=3, num_classes=1, base_filters=64):
        super(M2SNet, self).__init__()

        # 编码器
        self.encoder = Encoder(in_channels)

        # 多尺度相减模块 (保持与编码器相同的通道数)
        self.mss4 = MultiScaleSubtraction(512, 512)  # 512 -> 512
        self.mss3 = MultiScaleSubtraction(256, 256)  # 256 -> 256
        self.mss2 = MultiScaleSubtraction(128, 128)  # 128 -> 128
        self.mss1 = MultiScaleSubtraction(64, 64)  # 64 -> 64

        # 双向融合解码器
        # bfd4: 融合m4(512)和m3(256)，输出256
        self.bfd4 = BidirectionalFusionDecoder(512, 256, 256)
        # bfd3: 融合d4(256)和m2(128)，输出128
        self.bfd3 = BidirectionalFusionDecoder(256, 128, 128)
        # bfd2: 融合d3(128)和m1(64)，输出64
        self.bfd2 = BidirectionalFusionDecoder(128, 64, 64)

        # 最终上采样和输出
        self.final_conv = nn.Sequential(
            ConvBNReLU(64, 32, kernel_size=3, padding=1),
            ConvBNReLU(32, 32, kernel_size=3, padding=1),
        )
        self.out_conv = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x):
        input_size = x.size()[2:]

        # 编码
        x1, x2, x3, x4 = self.encoder(x)
        # x1: (B, 64, H/4, W/4)
        # x2: (B, 128, H/8, W/8)
        # x3: (B, 256, H/16, W/16)
        # x4: (B, 512, H/32, W/32)

        # MSS增强
        m4 = self.mss4(x4)  # (B, 512, H/32, W/32)
        m3 = self.mss3(x3)  # (B, 256, H/16, W/16)
        m2 = self.mss2(x2)  # (B, 128, H/8, W/8)
        m1 = self.mss1(x1)  # (B, 64, H/4, W/4)

        # BFD解码
        d4 = self.bfd4(m4, m3)  # (B, 256, H/16, W/16)
        d3 = self.bfd3(d4, m2)  # (B, 128, H/8, W/8)
        d2 = self.bfd2(d3, m1)  # (B, 64, H/4, W/4)

        # 最终处理
        out = self.final_conv(d2)
        out = self.out_conv(out)

        # 上采样到原始尺寸
        out = F.interpolate(out, size=input_size, mode='bilinear', align_corners=True)

        return out


# 测试代码
if __name__ == "__main__":
    print("=" * 60)
    print("M2SNet 模型测试")
    print("=" * 60)

    model = M2SNet(in_channels=3, num_classes=1, base_filters=64)

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

        # MSS
        m4 = model.mss4(x4)
        m3 = model.mss3(x3)
        m2 = model.mss2(x2)
        m1 = model.mss1(x1)
        print(f"MSS输出: m1={m1.shape}, m2={m2.shape}, m3={m3.shape}, m4={m4.shape}")

        # 完整前向传播
        y = model(x)

    print(f"最终输出尺寸: {y.shape}")

    assert y.shape == (2, 1, 352, 352), "输出尺寸错误！"
    print("\n✅ M2SNet 测试通过！")
    print("=" * 60)