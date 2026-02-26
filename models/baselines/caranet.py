"""
文件名: models/baselines/caranet.py
功能: CaraNet (Lou et al., 2022)
论文: CaraNet: Context Axial Reverse Attention Network for Segmentation of Small Medical Objects
会议: SPIE Medical Imaging 2022
引用: 200+
特点: 上下文轴向注意力 + 通道特征重校准(CFR)，专门针对小目标分割
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


class AxialAttention(nn.Module):
    """
    轴向注意力模块

    分别在水平和垂直方向计算注意力，降低计算复杂度
    """

    def __init__(self, in_channels, heads=8):
        super(AxialAttention, self).__init__()
        self.heads = heads
        self.head_dim = in_channels // heads

        self.query = nn.Conv2d(in_channels, in_channels, 1)
        self.key = nn.Conv2d(in_channels, in_channels, 1)
        self.value = nn.Conv2d(in_channels, in_channels, 1)

        self.out = nn.Conv2d(in_channels, in_channels, 1)
        self.scale = self.head_dim ** -0.5

    def forward(self, x):
        B, C, H, W = x.shape

        # 水平方向注意力
        q = self.query(x).view(B, self.heads, self.head_dim, H, W)
        k = self.key(x).view(B, self.heads, self.head_dim, H, W)
        v = self.value(x).view(B, self.heads, self.head_dim, H, W)

        # 沿宽度维度计算注意力
        q_h = q.permute(0, 1, 3, 2, 4).reshape(B * self.heads * H, self.head_dim, W)
        k_h = k.permute(0, 1, 3, 2, 4).reshape(B * self.heads * H, self.head_dim, W)
        v_h = v.permute(0, 1, 3, 2, 4).reshape(B * self.heads * H, self.head_dim, W)

        attn_h = torch.bmm(q_h.transpose(-2, -1), k_h) * self.scale
        attn_h = F.softmax(attn_h, dim=-1)
        out_h = torch.bmm(v_h, attn_h.transpose(-2, -1))
        out_h = out_h.reshape(B, self.heads, H, self.head_dim, W).permute(0, 1, 3, 2, 4)

        # 沿高度维度计算注意力
        q_w = q.permute(0, 1, 4, 2, 3).reshape(B * self.heads * W, self.head_dim, H)
        k_w = k.permute(0, 1, 4, 2, 3).reshape(B * self.heads * W, self.head_dim, H)
        v_w = v.permute(0, 1, 4, 2, 3).reshape(B * self.heads * W, self.head_dim, H)

        attn_w = torch.bmm(q_w.transpose(-2, -1), k_w) * self.scale
        attn_w = F.softmax(attn_w, dim=-1)
        out_w = torch.bmm(v_w, attn_w.transpose(-2, -1))
        out_w = out_w.reshape(B, self.heads, W, self.head_dim, H).permute(0, 1, 3, 4, 2)

        # 融合
        out = out_h + out_w
        out = out.reshape(B, C, H, W)
        out = self.out(out)

        return out + x


class ChannelFeatureRecalibration(nn.Module):
    """
    通道特征重校准模块 (Channel Feature Recalibration)

    对通道特征进行自适应重校准
    """

    def __init__(self, in_channels, reduction=16):
        super(ChannelFeatureRecalibration, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        att = self.sigmoid(avg_out + max_out)
        return x * att


class ContextAxialReverseAttention(nn.Module):
    """
    上下文轴向反向注意力模块

    结合轴向注意力、通道重校准和反向注意力
    """

    def __init__(self, in_channels, out_channels):
        super(ContextAxialReverseAttention, self).__init__()

        # 通道调整
        self.conv_in = ConvBNReLU(in_channels, out_channels, 1, padding=0)

        # 轴向注意力
        self.axial_att = AxialAttention(out_channels, heads=4)

        # 通道重校准
        self.cfr = ChannelFeatureRecalibration(out_channels)

        # 输出卷积
        self.conv_out = nn.Sequential(
            ConvBNReLU(out_channels, out_channels, 3, padding=1),
            nn.Conv2d(out_channels, 1, 1)
        )

    def forward(self, x, guide_map=None):
        x = self.conv_in(x)

        # 反向注意力引导
        if guide_map is not None:
            guide_map = F.interpolate(guide_map, size=x.size()[2:], mode='bilinear', align_corners=True)
            reverse_att = 1 - torch.sigmoid(guide_map)
            x = x * reverse_att + x

        # 轴向注意力
        x = self.axial_att(x)

        # 通道重校准
        x = self.cfr(x)

        # 输出
        out = self.conv_out(x)

        return out, x


class Encoder(nn.Module):
    """编码器"""

    def __init__(self, in_channels=3):
        super(Encoder, self).__init__()

        self.stem = nn.Sequential(
            ConvBNReLU(in_channels, 64, 7, stride=2, padding=3),
            nn.MaxPool2d(3, stride=2, padding=1)
        )

        self.layer1 = self._make_layer(64, 64, 3)
        self.layer2 = self._make_layer(64, 128, 4, stride=2)
        self.layer3 = self._make_layer(128, 256, 6, stride=2)
        self.layer4 = self._make_layer(256, 512, 3, stride=2)

    def _make_layer(self, in_ch, out_ch, blocks, stride=1):
        layers = [ConvBNReLU(in_ch, out_ch, 3, stride=stride, padding=1)]
        for _ in range(1, blocks):
            layers.append(ConvBNReLU(out_ch, out_ch, 3, padding=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        return x1, x2, x3, x4


class CaraNet(nn.Module):
    """
    CaraNet: Context Axial Reverse Attention Network

    架构：
    1. ResNet风格编码器
    2. 多级CARA模块
    3. 渐进式细化解码

    参数:
        in_channels: 输入通道数 (RGB=3)
        num_classes: 输出类别数
        base_filters: 基础滤波器数量
    """

    def __init__(self, in_channels=3, num_classes=1, base_filters=64):
        super(CaraNet, self).__init__()

        # 编码器
        self.encoder = Encoder(in_channels)

        # CARA模块
        self.cara4 = ContextAxialReverseAttention(512, 256)
        self.cara3 = ContextAxialReverseAttention(256, 128)
        self.cara2 = ContextAxialReverseAttention(128, 64)
        self.cara1 = ContextAxialReverseAttention(64, 32)

        # 上采样
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x):
        input_size = x.size()[2:]

        # 编码
        x1, x2, x3, x4 = self.encoder(x)

        # CARA解码
        out4, feat4 = self.cara4(x4)
        out3, feat3 = self.cara3(x3, out4)
        out2, feat2 = self.cara2(x2, out3)
        out1, feat1 = self.cara1(x1, out2)

        # 上采样到原始尺寸
        out = F.interpolate(out1, size=input_size, mode='bilinear', align_corners=True)

        return out


# 测试代码
if __name__ == "__main__":
    print("=" * 60)
    print("CaraNet 模型测试")
    print("=" * 60)

    model = CaraNet(in_channels=3, num_classes=1, base_filters=64)

    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n参数量: {total_params / 1e6:.2f}M")

    # 测试前向传播
    x = torch.randn(2, 3, 352, 352)
    y = model(x)

    print(f"输入尺寸: {x.shape}")
    print(f"输出尺寸: {y.shape}")

    assert y.shape == (2, 1, 352, 352), "输出尺寸错误！"
    print("\n✅ CaraNet 测试通过！")
    print("=" * 60)