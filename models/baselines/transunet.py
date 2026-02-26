"""
文件名: models/baselines/transunet.py
功能: TransUNet简化版 (Chen et al., 2021)
论文: TransUNet: Transformers Make Strong Encoders for Medical Image Segmentation
说明: 简化版本，不依赖预训练ViT
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ConvBlock(nn.Module):
    """双卷积块"""

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


class MultiHeadAttention(nn.Module):
    """多头自注意力"""

    def __init__(self, embed_dim=512, num_heads=8, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

        self.scale = self.head_dim ** -0.5

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer编码器块"""

    def __init__(self, embed_dim=512, num_heads=8, mlp_ratio=4.0, dropout=0.1):
        super(TransformerBlock, self).__init__()

        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)

        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_hidden = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class TransUNet(nn.Module):
    """TransUNet: CNN + Transformer混合架构"""

    def __init__(self, in_channels=1, num_classes=1, base_filters=64,
                 embed_dim=256, num_heads=8, num_layers=4, patch_size=2):
        super(TransUNet, self).__init__()

        f = [base_filters * (2 ** i) for i in range(5)]
        self.patch_size = patch_size

        # CNN编码器
        self.enc1 = ConvBlock(in_channels, f[0])
        self.enc2 = ConvBlock(f[0], f[1])
        self.enc3 = ConvBlock(f[1], f[2])
        self.enc4 = ConvBlock(f[2], f[3])

        self.pool = nn.MaxPool2d(2)

        # Patch Embedding
        self.patch_embed = nn.Conv2d(f[3], embed_dim, kernel_size=patch_size, stride=patch_size)

        # 位置编码 (支持不同输入尺寸)
        self.pos_embed = nn.Parameter(torch.zeros(1, 256, embed_dim))  # 最大256个patch
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Transformer
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads) for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(embed_dim)

        # 从Transformer恢复到CNN特征
        self.proj_back = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, f[4], kernel_size=patch_size, stride=patch_size),
            nn.BatchNorm2d(f[4]),
            nn.ReLU(inplace=True)
        )

        # CNN解码器
        self.up4 = nn.ConvTranspose2d(f[4], f[3], kernel_size=2, stride=2)
        self.dec4 = ConvBlock(f[4], f[3])

        self.up3 = nn.ConvTranspose2d(f[3], f[2], kernel_size=2, stride=2)
        self.dec3 = ConvBlock(f[3], f[2])

        self.up2 = nn.ConvTranspose2d(f[2], f[1], kernel_size=2, stride=2)
        self.dec2 = ConvBlock(f[2], f[1])

        self.up1 = nn.ConvTranspose2d(f[1], f[0], kernel_size=2, stride=2)
        self.dec1 = ConvBlock(f[1], f[0])

        # 输出
        self.out = nn.Conv2d(f[0], num_classes, kernel_size=1)

    def forward(self, x):
        # CNN编码
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        e4_pool = self.pool(e4)

        # Patch Embedding
        B, C, H, W = e4_pool.shape
        tokens = self.patch_embed(e4_pool)
        tokens = tokens.flatten(2).transpose(1, 2)

        # 添加位置编码
        num_patches = tokens.size(1)
        tokens = tokens + self.pos_embed[:, :num_patches, :]

        # Transformer
        for block in self.transformer_blocks:
            tokens = block(tokens)

        tokens = self.norm(tokens)

        # 恢复空间维度
        H_out = H // self.patch_size
        W_out = W // self.patch_size
        tokens = tokens.transpose(1, 2).reshape(B, -1, H_out, W_out)

        # 投影回CNN特征空间
        b = self.proj_back(tokens)

        # CNN解码
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return self.out(d1)


if __name__ == "__main__":
    model = TransUNet(in_channels=1, num_classes=1, embed_dim=256, num_heads=8, num_layers=4)
    x = torch.randn(2, 1, 512, 512)
    y = model(x)
    print(f"Input: {x.shape}, Output: {y.shape}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")