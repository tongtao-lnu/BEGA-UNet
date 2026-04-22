"""
文件名: models/ega_unet.py
功能: EGA-UNet模型定义
说明: 融合边缘引导、双路径注意力和多尺度特征聚合的医学图像分割网络
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """
    基础卷积块: Conv -> BN -> ReLU -> Conv -> BN -> ReLU
    U-Net基本构建单元
    """

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


class EdgeGuidedModule(nn.Module):
    """
    边缘引导模块 (Edge-Guided Module, EGM)

    创新点1：显式提取边缘特征并融合到主干网络

    工作原理：
    1. 使用可学习的Sobel算子提取水平和垂直边缘
    2. 通过卷积层增强边缘特征
    3. 使用注意力机制将边缘信息融合到主特征中
    """

    def __init__(self, channels):
        super(EdgeGuidedModule, self).__init__()

        # 可学习的Sobel边缘检测
        self.edge_conv_x = nn.Conv2d(channels, channels, kernel_size=3,
                                     padding=1, groups=channels, bias=False)
        self.edge_conv_y = nn.Conv2d(channels, channels, kernel_size=3,
                                     padding=1, groups=channels, bias=False)

        # 初始化Sobel核权重
        sobel_x = torch.tensor([[-1, 0, 1],
                                [-2, 0, 2],
                                [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1],
                                [0, 0, 0],
                                [1, 2, 1]], dtype=torch.float32)

        self.edge_conv_x.weight.data = sobel_x.view(1, 1, 3, 3).repeat(channels, 1, 1, 1)
        self.edge_conv_y.weight.data = sobel_y.view(1, 1, 3, 3).repeat(channels, 1, 1, 1)

        # 边缘特征增强
        self.edge_enhance = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

        # 注意力融合门
        self.fusion_gate = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        edge_x = self.edge_conv_x(x)
        edge_y = self.edge_conv_y(x)

        edge_combined = torch.cat([edge_x, edge_y], dim=1)
        edge_feat = self.edge_enhance(edge_combined)

        fusion_input = torch.cat([x, edge_feat], dim=1)
        attention = self.fusion_gate(fusion_input)

        enhanced_feat = x * (1 + attention) + edge_feat * attention

        return enhanced_feat, edge_feat


class DualPathAttention(nn.Module):
    """
    双路径注意力机制 (Dual-Path Attention, DPA)

    创新点2：并行处理通道和空间注意力，提高效率
    """

    def __init__(self, channels, reduction=16):
        super(DualPathAttention, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        reduced_channels = max(channels // reduction, 8)
        self.channel_fc = nn.Sequential(
            nn.Linear(channels, reduced_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_channels, channels, bias=False)
        )

        self.spatial_conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.fusion = nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(channels)

    def forward(self, x):
        B, C, H, W = x.size()

        # 通道注意力
        avg_out = self.avg_pool(x).view(B, C)
        max_out = self.max_pool(x).view(B, C)

        channel_att = torch.sigmoid(
            self.channel_fc(avg_out) + self.channel_fc(max_out)
        ).view(B, C, 1, 1)

        channel_refined = x * channel_att

        # 空间注意力
        avg_spatial = torch.mean(x, dim=1, keepdim=True)
        max_spatial, _ = torch.max(x, dim=1, keepdim=True)

        spatial_input = torch.cat([avg_spatial, max_spatial], dim=1)
        spatial_att = self.spatial_conv(spatial_input)

        spatial_refined = x * spatial_att

        # 融合
        combined = torch.cat([channel_refined, spatial_refined], dim=1)
        out = self.bn(self.fusion(combined))

        return out + x


class MultiScaleFeatureAggregation(nn.Module):
    """多尺度特征聚合模块 (MSFA) - 添加中间输出版本"""

    def __init__(self, in_channels, out_channels):
        super(MultiScaleFeatureAggregation, self).__init__()

        branch_channels = in_channels // 4

        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True)
        )

        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=3,
                      padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True)
        )

        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=3,
                      padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True)
        )

        self.branch4 = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=3,
                      padding=4, dilation=4, bias=False),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True)
        )

        self.global_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True)
        )

        self.fusion = nn.Sequential(
            nn.Conv2d(branch_channels * 5, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, return_branches=False):
        """
        前向传播

        Args:
            x: 输入特征
            return_branches: 是否返回各分支的中间特征（用于可视化）
        """
        size = x.size()[2:]

        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)

        b5 = self.global_branch(x)
        b5 = F.interpolate(b5, size=size, mode='bilinear', align_corners=True)

        out = torch.cat([b1, b2, b3, b4, b5], dim=1)
        out = self.fusion(out)

        if return_branches:
            branch_features = {
                'branch_1x1': b1,
                'branch_d1': b2,
                'branch_d2': b3,
                'branch_d4': b4,
                'branch_global': b5,
                'fused': out
            }
            return out, branch_features

        return out

class EncoderBlock(nn.Module):
    """编码器块"""

    def __init__(self, in_channels, out_channels, use_egm=True, use_dpa=True):
        super(EncoderBlock, self).__init__()

        self.use_egm = use_egm
        self.use_dpa = use_dpa

        self.conv = ConvBlock(in_channels, out_channels)

        if use_egm:
            self.egm = EdgeGuidedModule(out_channels)

        if use_dpa:
            self.dpa = DualPathAttention(out_channels)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(x)

        edge_feat = None
        if self.use_egm:
            x, edge_feat = self.egm(x)

        if self.use_dpa:
            x = self.dpa(x)

        pooled = self.pool(x)

        return pooled, x, edge_feat


class DecoderBlock(nn.Module):
    """解码器块"""

    def __init__(self, in_channels, skip_channels, out_channels, use_dpa=True):
        super(DecoderBlock, self).__init__()

        self.use_dpa = use_dpa

        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2,
                                     kernel_size=2, stride=2)

        self.conv = ConvBlock(in_channels // 2 + skip_channels, out_channels)

        if use_dpa:
            self.dpa = DualPathAttention(out_channels)

    def forward(self, x, skip):
        x = self.up(x)

        if x.size() != skip.size():
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)

        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)

        if self.use_dpa:
            x = self.dpa(x)

        return x


class EGAUNet(nn.Module):
    """
    EGA-UNet: Edge-Guided Attention U-Net
    """

    def __init__(self, in_channels=1, num_classes=1, base_filters=64):
        super(EGAUNet, self).__init__()

        f = [base_filters * (2 ** i) for i in range(5)]

        # 编码器
        self.enc1 = EncoderBlock(in_channels, f[0], use_egm=True, use_dpa=False)
        self.enc2 = EncoderBlock(f[0], f[1], use_egm=True, use_dpa=True)
        self.enc3 = EncoderBlock(f[1], f[2], use_egm=True, use_dpa=True)
        self.enc4 = EncoderBlock(f[2], f[3], use_egm=True, use_dpa=True)

        # 瓶颈层
        self.bottleneck_conv = ConvBlock(f[3], f[4])
        self.msfa = MultiScaleFeatureAggregation(f[4], f[4])

        # 解码器
        self.dec4 = DecoderBlock(f[4], f[3], f[3], use_dpa=True)
        self.dec3 = DecoderBlock(f[3], f[2], f[2], use_dpa=True)
        self.dec2 = DecoderBlock(f[2], f[1], f[1], use_dpa=True)
        self.dec1 = DecoderBlock(f[1], f[0], f[0], use_dpa=False)

        # 输出层
        self.seg_out = nn.Conv2d(f[0], num_classes, kernel_size=1)
        self.edge_out = nn.Conv2d(f[0], 1, kernel_size=1)

    def forward(self, x):
        # 编码
        p1, s1, e1 = self.enc1(x)
        p2, s2, e2 = self.enc2(p1)
        p3, s3, e3 = self.enc3(p2)
        p4, s4, e4 = self.enc4(p3)

        # 瓶颈
        b = self.bottleneck_conv(p4)
        b = self.msfa(b)

        # 解码
        d4 = self.dec4(b, s4)
        d3 = self.dec3(d4, s3)
        d2 = self.dec2(d3, s2)
        d1 = self.dec1(d2, s1)

        # 输出
        seg_output = self.seg_out(d1)
        edge_output = self.edge_out(d1)

        edge_features = [e for e in [e1, e2, e3, e4] if e is not None]

        return seg_output, edge_output, edge_features


def test_model():
    """测试模型"""
    print("=" * 60)
    print("EGA-UNet 模型测试")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"测试设备: {device}")

    model = EGAUNet(in_channels=1, num_classes=1, base_filters=64)
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"总参数量: {total_params / 1e6:.2f}M")

    x = torch.randn(2, 1, 512, 512).to(device)

    with torch.no_grad():
        seg_out, edge_out, edge_feats = model(x)

    print(f"输入尺寸: {x.shape}")
    print(f"分割输出尺寸: {seg_out.shape}")
    print(f"边缘输出尺寸: {edge_out.shape}")

    print("=" * 60)
    print("✅ 模型测试通过！")
    print("=" * 60)


if __name__ == "__main__":
    test_model()
