"""
文件名: models/ablation_models.py
功能: 消融实验模型变体定义
说明: 用于验证EGA-UNet各组件(EGM, DPA, MSFA)的有效性
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """基础卷积块"""

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
    """边缘引导模块 (EGM)"""

    def __init__(self, channels):
        super(EdgeGuidedModule, self).__init__()

        self.edge_conv_x = nn.Conv2d(channels, channels, kernel_size=3,
                                     padding=1, groups=channels, bias=False)
        self.edge_conv_y = nn.Conv2d(channels, channels, kernel_size=3,
                                     padding=1, groups=channels, bias=False)

        # 初始化Sobel核
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)

        self.edge_conv_x.weight.data = sobel_x.view(1, 1, 3, 3).repeat(channels, 1, 1, 1)
        self.edge_conv_y.weight.data = sobel_y.view(1, 1, 3, 3).repeat(channels, 1, 1, 1)

        self.edge_enhance = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

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
    """双路径注意力机制 (DPA)"""

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
    """多尺度特征聚合模块 (MSFA)"""

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

    def forward(self, x):
        size = x.size()[2:]
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)
        b5 = self.global_branch(x)
        b5 = F.interpolate(b5, size=size, mode='bilinear', align_corners=True)
        out = torch.cat([b1, b2, b3, b4, b5], dim=1)
        out = self.fusion(out)
        return out


class AblationUNet(nn.Module):
    """
    可配置的消融实验U-Net

    参数:
        use_egm: 是否使用边缘引导模块
        use_dpa: 是否使用双路径注意力
        use_msfa: 是否使用多尺度特征聚合
    """

    def __init__(self, in_channels=1, num_classes=1, base_filters=64,
                 use_egm=False, use_dpa=False, use_msfa=False):
        super(AblationUNet, self).__init__()

        self.use_egm = use_egm
        self.use_dpa = use_dpa
        self.use_msfa = use_msfa

        f = [base_filters * (2 ** i) for i in range(5)]

        # 编码器
        self.enc1 = ConvBlock(in_channels, f[0])
        self.enc2 = ConvBlock(f[0], f[1])
        self.enc3 = ConvBlock(f[1], f[2])
        self.enc4 = ConvBlock(f[2], f[3])

        self.pool = nn.MaxPool2d(2)

        # EGM模块（如果启用）
        if use_egm:
            self.egm1 = EdgeGuidedModule(f[0])
            self.egm2 = EdgeGuidedModule(f[1])
            self.egm3 = EdgeGuidedModule(f[2])
            self.egm4 = EdgeGuidedModule(f[3])

        # DPA模块（如果启用）
        if use_dpa:
            self.dpa1 = DualPathAttention(f[0])
            self.dpa2 = DualPathAttention(f[1])
            self.dpa3 = DualPathAttention(f[2])
            self.dpa4 = DualPathAttention(f[3])

        # 瓶颈层
        self.bottleneck = ConvBlock(f[3], f[4])

        # MSFA模块（如果启用）
        if use_msfa:
            self.msfa = MultiScaleFeatureAggregation(f[4], f[4])

        # 解码器
        self.up4 = nn.ConvTranspose2d(f[4], f[3], kernel_size=2, stride=2)
        self.dec4 = ConvBlock(f[4], f[3])

        self.up3 = nn.ConvTranspose2d(f[3], f[2], kernel_size=2, stride=2)
        self.dec3 = ConvBlock(f[3], f[2])

        self.up2 = nn.ConvTranspose2d(f[2], f[1], kernel_size=2, stride=2)
        self.dec2 = ConvBlock(f[2], f[1])

        self.up1 = nn.ConvTranspose2d(f[1], f[0], kernel_size=2, stride=2)
        self.dec1 = ConvBlock(f[1], f[0])

        # 解码器DPA（如果启用）
        if use_dpa:
            self.dec_dpa4 = DualPathAttention(f[3])
            self.dec_dpa3 = DualPathAttention(f[2])
            self.dec_dpa2 = DualPathAttention(f[1])

        # 输出层
        self.seg_out = nn.Conv2d(f[0], num_classes, kernel_size=1)

        # 边缘输出（如果使用EGM）
        if use_egm:
            self.edge_out = nn.Conv2d(f[0], 1, kernel_size=1)

    def forward(self, x):
        # 编码器
        e1 = self.enc1(x)
        if self.use_egm:
            e1, edge1 = self.egm1(e1)
        if self.use_dpa:
            e1 = self.dpa1(e1)

        e2 = self.enc2(self.pool(e1))
        if self.use_egm:
            e2, edge2 = self.egm2(e2)
        if self.use_dpa:
            e2 = self.dpa2(e2)

        e3 = self.enc3(self.pool(e2))
        if self.use_egm:
            e3, edge3 = self.egm3(e3)
        if self.use_dpa:
            e3 = self.dpa3(e3)

        e4 = self.enc4(self.pool(e3))
        if self.use_egm:
            e4, edge4 = self.egm4(e4)
        if self.use_dpa:
            e4 = self.dpa4(e4)

        # 瓶颈
        b = self.bottleneck(self.pool(e4))
        if self.use_msfa:
            b = self.msfa(b)

        # 解码器
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        if self.use_dpa:
            d4 = self.dec_dpa4(d4)

        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        if self.use_dpa:
            d3 = self.dec_dpa3(d3)

        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        if self.use_dpa:
            d2 = self.dec_dpa2(d2)

        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        # 输出
        seg_output = self.seg_out(d1)

        if self.use_egm:
            edge_output = self.edge_out(d1)
            edge_features = [edge1, edge2, edge3, edge4]
            return seg_output, edge_output, edge_features
        else:
            return seg_output


# ============================================================
# 消融实验配置字典
# ============================================================

ABLATION_CONFIGS = {
    'baseline': {
        'use_egm': False,
        'use_dpa': False,
        'use_msfa': False,
        'description': 'Baseline U-Net'
    },
    'egm_only': {
        'use_egm': True,
        'use_dpa': False,
        'use_msfa': False,
        'description': 'U-Net + EGM'
    },
    'dpa_only': {
        'use_egm': False,
        'use_dpa': True,
        'use_msfa': False,
        'description': 'U-Net + DPA'
    },
    'msfa_only': {
        'use_egm': False,
        'use_dpa': False,
        'use_msfa': True,
        'description': 'U-Net + MSFA'
    },
    'egm_dpa': {
        'use_egm': True,
        'use_dpa': True,
        'use_msfa': False,
        'description': 'U-Net + EGM + DPA'
    },
    'egm_msfa': {
        'use_egm': True,
        'use_dpa': False,
        'use_msfa': True,
        'description': 'U-Net + EGM + MSFA'
    },
    'dpa_msfa': {
        'use_egm': False,
        'use_dpa': True,
        'use_msfa': True,
        'description': 'U-Net + DPA + MSFA'
    },
}


def get_ablation_model(config_name, in_channels=3, num_classes=1, base_filters=64):
    """根据配置名称获取消融实验模型

    参数:
        config_name: 配置名称
        in_channels: 输入通道数（RGB图像为3）
        num_classes: 输出类别数
        base_filters: 基础滤波器数量
    """
    if config_name not in ABLATION_CONFIGS:
        raise ValueError(f"未知配置: {config_name}. 可用配置: {list(ABLATION_CONFIGS.keys())}")

    config = ABLATION_CONFIGS[config_name]
    model = AblationUNet(
        in_channels=in_channels,
        num_classes=num_classes,
        base_filters=base_filters,
        use_egm=config['use_egm'],
        use_dpa=config['use_dpa'],
        use_msfa=config['use_msfa']
    )

    return model, config


def test_ablation_models():
    """测试所有消融实验模型"""
    print("=" * 60)
    print("消融实验模型测试")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = torch.randn(2, 1, 512, 512).to(device)

    for config_name in ABLATION_CONFIGS.keys():
        model, config = get_ablation_model(config_name)
        model = model.to(device)

        params = sum(p.numel() for p in model.parameters()) / 1e6

        with torch.no_grad():
            output = model(x)

        if config['use_egm']:
            seg_out, edge_out, _ = output
            out_shape = seg_out.shape
        else:
            out_shape = output.shape

        print(f"✅ {config_name:12s} | {config['description']:20s} | "
              f"Params: {params:.2f}M | Output: {out_shape}")

    print("=" * 60)
    print("所有消融模型测试通过！")


if __name__ == "__main__":
    test_ablation_models()