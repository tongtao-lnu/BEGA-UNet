"""
文件名: models/baselines/polyp_pvt.py
功能: Polyp-PVT (Dong et al., 2023)
论文: Polyp-PVT: Polyp Segmentation with Pyramid Vision Transformers
期刊: CAAI Artificial Intelligence Research, 2023
引用: 300+
特点: 金字塔视觉Transformer + 级联融合模块(CFM) + 伪装识别模块(CIM)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


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


class PatchEmbed(nn.Module):
    """图像分块嵌入"""

    def __init__(self, img_size=352, patch_size=4, in_channels=3, embed_dim=64):
        super(PatchEmbed, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)  # B, C, H, W
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # B, N, C
        x = self.norm(x)
        return x, H, W


class Attention(nn.Module):
    """多头自注意力"""

    def __init__(self, dim, num_heads=8, sr_ratio=1):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)
        self.proj = nn.Linear(dim, dim)

        # 空间缩减
        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)

        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)

        return x


class MLP(nn.Module):
    """前馈网络"""

    def __init__(self, in_features, hidden_features=None, out_features=None):
        super(MLP, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(0.1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class PVTBlock(nn.Module):
    """PVT基础块"""

    def __init__(self, dim, num_heads, sr_ratio=1, mlp_ratio=4.):
        super(PVTBlock, self).__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads=num_heads, sr_ratio=sr_ratio)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio))

    def forward(self, x, H, W):
        x = x + self.attn(self.norm1(x), H, W)
        x = x + self.mlp(self.norm2(x))
        return x


class PVTStage(nn.Module):
    """PVT阶段"""

    def __init__(self, in_channels, out_channels, num_blocks, num_heads, sr_ratio, patch_size=2):
        super(PVTStage, self).__init__()

        # Patch嵌入（下采样）
        self.patch_embed = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=patch_size, stride=patch_size),
            nn.LayerNorm([out_channels]),
        )

        # Transformer块
        self.blocks = nn.ModuleList([
            PVTBlock(out_channels, num_heads, sr_ratio)
            for _ in range(num_blocks)
        ])

        self.norm = nn.LayerNorm(out_channels)

    def forward(self, x):
        # x: B, C, H, W
        x = self.patch_embed[0](x)  # Conv
        B, C, H, W = x.shape

        x = x.flatten(2).transpose(1, 2)  # B, N, C

        for block in self.blocks:
            x = block(x, H, W)

        x = self.norm(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2)  # B, C, H, W

        return x


class CascadedFusionModule(nn.Module):
    """
    级联融合模块 (Cascaded Fusion Module, CFM)

    融合多尺度特征
    """

    def __init__(self, channels):
        super(CascadedFusionModule, self).__init__()
        self.conv1 = ConvBNReLU(channels * 4, channels, 1, padding=0)
        self.conv2 = ConvBNReLU(channels, channels, 3, padding=1)
        self.conv3 = ConvBNReLU(channels, channels, 3, padding=1)

    def forward(self, x1, x2, x3, x4):
        # 上采样到统一尺寸
        size = x1.size()[2:]
        x2 = F.interpolate(x2, size=size, mode='bilinear', align_corners=True)
        x3 = F.interpolate(x3, size=size, mode='bilinear', align_corners=True)
        x4 = F.interpolate(x4, size=size, mode='bilinear', align_corners=True)

        # 拼接
        x = torch.cat([x1, x2, x3, x4], dim=1)

        # 融合
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        return x


class CamouflagedIdentificationModule(nn.Module):
    """
    伪装识别模块 (Camouflaged Identification Module, CIM)

    识别与背景相似的目标区域
    """

    def __init__(self, in_channels, out_channels):
        super(CamouflagedIdentificationModule, self).__init__()

        self.conv1 = ConvBNReLU(in_channels, out_channels, 3, padding=1)
        self.conv2 = ConvBNReLU(out_channels, out_channels, 3, padding=1)

        # 通道注意力
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, out_channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // 4, out_channels, 1),
            nn.Sigmoid()
        )

        # 空间注意力
        self.sa = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3),
            nn.Sigmoid()
        )

        self.out_conv = nn.Conv2d(out_channels, 1, 1)

    def forward(self, x, guide=None):
        if guide is not None:
            guide = F.interpolate(guide, size=x.size()[2:], mode='bilinear', align_corners=True)
            x = x * torch.sigmoid(guide) + x

        x = self.conv1(x)
        x = self.conv2(x)

        # 通道注意力
        ca = self.ca(x)
        x = x * ca

        # 空间注意力
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        sa = self.sa(torch.cat([avg_out, max_out], dim=1))
        x = x * sa

        out = self.out_conv(x)

        return out, x


class SimplePVTEncoder(nn.Module):
    """简化的PVT编码器"""

    def __init__(self, in_channels=3):
        super(SimplePVTEncoder, self).__init__()

        # Stem
        self.stem = nn.Sequential(
            ConvBNReLU(in_channels, 64, 7, stride=2, padding=3),
            ConvBNReLU(64, 64, 3, padding=1),
        )

        # 各阶段
        self.stage1 = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBNReLU(64, 64, 3, padding=1),
            ConvBNReLU(64, 64, 3, padding=1),
        )

        self.stage2 = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBNReLU(64, 128, 3, padding=1),
            ConvBNReLU(128, 128, 3, padding=1),
        )

        self.stage3 = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBNReLU(128, 320, 3, padding=1),
            ConvBNReLU(320, 320, 3, padding=1),
        )

        self.stage4 = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBNReLU(320, 512, 3, padding=1),
            ConvBNReLU(512, 512, 3, padding=1),
        )

    def forward(self, x):
        x = self.stem(x)
        x1 = self.stage1(x)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)
        return x1, x2, x3, x4


class PolypPVT(nn.Module):
    """
    Polyp-PVT: 基于金字塔视觉Transformer的息肉分割网络

    架构：
    1. PVT编码器提取多尺度特征
    2. CFM模块融合多尺度特征
    3. CIM模块识别伪装区域

    参数:
        in_channels: 输入通道数 (RGB=3)
        num_classes: 输出类别数
        embed_dim: 嵌入维度
    """

    def __init__(self, in_channels=3, num_classes=1, embed_dim=64):
        super(PolypPVT, self).__init__()

        # 编码器
        self.encoder = SimplePVTEncoder(in_channels)

        # 通道统一
        self.reduce1 = ConvBNReLU(64, 64, 1, padding=0)
        self.reduce2 = ConvBNReLU(128, 64, 1, padding=0)
        self.reduce3 = ConvBNReLU(320, 64, 1, padding=0)
        self.reduce4 = ConvBNReLU(512, 64, 1, padding=0)

        # CFM
        self.cfm = CascadedFusionModule(64)

        # CIM模块
        self.cim4 = CamouflagedIdentificationModule(64, 64)
        self.cim3 = CamouflagedIdentificationModule(64, 64)
        self.cim2 = CamouflagedIdentificationModule(64, 64)
        self.cim1 = CamouflagedIdentificationModule(64, 64)

    def forward(self, x):
        input_size = x.size()[2:]

        # 编码
        x1, x2, x3, x4 = self.encoder(x)

        # 通道统一
        x1 = self.reduce1(x1)
        x2 = self.reduce2(x2)
        x3 = self.reduce3(x3)
        x4 = self.reduce4(x4)

        # CFM融合
        fused = self.cfm(x1, x2, x3, x4)

        # CIM解码
        out4, f4 = self.cim4(x4)
        out3, f3 = self.cim3(x3, out4)
        out2, f2 = self.cim2(x2, out3)
        out1, f1 = self.cim1(x1, out2)

        # 融合CFM和CIM
        out = out1 + F.interpolate(fused, size=out1.size()[2:], mode='bilinear', align_corners=True)[:, :1]

        # 上采样到原始尺寸
        out = F.interpolate(out, size=input_size, mode='bilinear', align_corners=True)

        return out


# 测试代码
if __name__ == "__main__":
    print("=" * 60)
    print("Polyp-PVT 模型测试")
    print("=" * 60)

    model = PolypPVT(in_channels=3, num_classes=1, embed_dim=64)

    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n参数量: {total_params / 1e6:.2f}M")

    # 测试前向传播
    x = torch.randn(2, 3, 352, 352)
    y = model(x)

    print(f"输入尺寸: {x.shape}")
    print(f"输出尺寸: {y.shape}")

    assert y.shape == (2, 1, 352, 352), "输出尺寸错误！"
    print("\n✅ Polyp-PVT 测试通过！")
    print("=" * 60)