"""
文件名: models/baselines/segnet.py
功能: SegNet (Badrinarayanan et al., 2017)
论文: SegNet: A Deep Convolutional Encoder-Decoder Architecture for Image Segmentation
期刊: IEEE TPAMI 2017
特点: 使用池化索引(pooling indices)进行非线性上采样，结构简洁高效
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


class EncoderBlock(nn.Module):
    """SegNet编码器块"""

    def __init__(self, in_channels, out_channels, num_convs=2):
        super(EncoderBlock, self).__init__()

        layers = []
        layers.append(ConvBNReLU(in_channels, out_channels))
        for _ in range(num_convs - 1):
            layers.append(ConvBNReLU(out_channels, out_channels))

        self.convs = nn.Sequential(*layers)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

    def forward(self, x):
        x = self.convs(x)
        x_pooled, indices = self.pool(x)
        return x_pooled, indices, x.size()


class DecoderBlock(nn.Module):
    """SegNet解码器块"""

    def __init__(self, in_channels, out_channels, num_convs=2):
        super(DecoderBlock, self).__init__()

        self.unpool = nn.MaxUnpool2d(kernel_size=2, stride=2)

        layers = []
        layers.append(ConvBNReLU(in_channels, in_channels))
        for _ in range(num_convs - 2):
            layers.append(ConvBNReLU(in_channels, in_channels))
        layers.append(ConvBNReLU(in_channels, out_channels))

        self.convs = nn.Sequential(*layers)

    def forward(self, x, indices, output_size):
        x = self.unpool(x, indices, output_size=output_size)
        x = self.convs(x)
        return x


class SegNet(nn.Module):
    """
    SegNet: A Deep Convolutional Encoder-Decoder Architecture

    特点:
    1. 编码器结构类似VGG16
    2. 使用最大池化索引进行上采样（不需要学习上采样参数）
    3. 解码器与编码器对称
    4. 结构简洁，训练速度快

    参数:
        in_channels: 输入通道数 (RGB=3)
        num_classes: 输出类别数
        base_filters: 基础滤波器数量
    """

    def __init__(self, in_channels=3, num_classes=1, base_filters=64):
        super(SegNet, self).__init__()

        f = base_filters  # 64

        # 编码器 (类似VGG结构)
        self.enc1 = EncoderBlock(in_channels, f, num_convs=2)  # 64
        self.enc2 = EncoderBlock(f, f * 2, num_convs=2)  # 128
        self.enc3 = EncoderBlock(f * 2, f * 4, num_convs=3)  # 256
        self.enc4 = EncoderBlock(f * 4, f * 8, num_convs=3)  # 512
        self.enc5 = EncoderBlock(f * 8, f * 8, num_convs=3)  # 512

        # 解码器 (与编码器对称)
        self.dec5 = DecoderBlock(f * 8, f * 8, num_convs=3)  # 512
        self.dec4 = DecoderBlock(f * 8, f * 4, num_convs=3)  # 256
        self.dec3 = DecoderBlock(f * 4, f * 2, num_convs=3)  # 128
        self.dec2 = DecoderBlock(f * 2, f, num_convs=2)  # 64
        self.dec1 = DecoderBlock(f, f, num_convs=2)  # 64

        # 输出层
        self.out_conv = nn.Conv2d(f, num_classes, kernel_size=1)

    def forward(self, x):
        # 编码器
        x, idx1, size1 = self.enc1(x)
        x, idx2, size2 = self.enc2(x)
        x, idx3, size3 = self.enc3(x)
        x, idx4, size4 = self.enc4(x)
        x, idx5, size5 = self.enc5(x)

        # 解码器
        x = self.dec5(x, idx5, size5)
        x = self.dec4(x, idx4, size4)
        x = self.dec3(x, idx3, size3)
        x = self.dec2(x, idx2, size2)
        x = self.dec1(x, idx1, size1)

        # 输出
        out = self.out_conv(x)

        return out


# 测试代码
if __name__ == "__main__":
    # 测试模型
    model = SegNet(in_channels=3, num_classes=1, base_filters=64)

    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"SegNet 参数量: {total_params / 1e6:.2f}M")

    # 测试前向传播
    x = torch.randn(2, 3, 352, 352)
    y = model(x)
    print(f"输入尺寸: {x.shape}")
    print(f"输出尺寸: {y.shape}")

    # 验证
    assert y.shape == (2, 1, 352, 352), "输出尺寸错误！"
    print("✅ SegNet 测试通过！")