"""
文件名: models/baselines/pranet.py
功能: PraNet (Fan et al., 2020)
论文: PraNet: Parallel Reverse Attention Network for Polyp Segmentation
会议: MICCAI 2020
引用: 1500+
特点: 并行部分解码器(PPD) + 反向注意力模块(RA)，专为息肉分割设计
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicConv2d(nn.Module):
    """基础卷积块：Conv + BN + ReLU"""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class RFB_modified(nn.Module):
    """
    改进的感受野模块 (Receptive Field Block)
    使用多尺度空洞卷积捕获不同感受野的特征
    """

    def __init__(self, in_channel, out_channel):
        super(RFB_modified, self).__init__()
        self.relu = nn.ReLU(True)

        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )

        self.conv_cat = BasicConv2d(4 * out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)

        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), dim=1))
        x = self.relu(x_cat + self.conv_res(x))
        return x


class PartialDecoder(nn.Module):
    """
    部分解码器聚合模块
    融合多层次特征
    """

    def __init__(self, channel):
        super(PartialDecoder, self).__init__()
        self.relu = nn.ReLU(True)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_upsample1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample3 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample4 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample5 = BasicConv2d(2 * channel, 2 * channel, 3, padding=1)

        self.conv_concat2 = BasicConv2d(2 * channel, 2 * channel, 3, padding=1)
        self.conv_concat3 = BasicConv2d(3 * channel, 3 * channel, 3, padding=1)
        self.conv4 = BasicConv2d(3 * channel, 3 * channel, 3, padding=1)
        self.conv5 = nn.Conv2d(3 * channel, 1, 1)

    def forward(self, x1, x2, x3):
        x1_1 = x1
        x2_1 = self.conv_upsample1(self.upsample(x1)) * x2
        x3_1 = self.conv_upsample2(self.upsample(x2_1)) * self.conv_upsample3(self.upsample(x2)) * x3

        x2_2 = torch.cat((x2_1, self.conv_upsample4(self.upsample(x1_1))), dim=1)
        x2_2 = self.conv_concat2(x2_2)

        x3_2 = torch.cat((x3_1, self.conv_upsample5(self.upsample(x2_2))), dim=1)
        x3_2 = self.conv_concat3(x3_2)

        x = self.conv4(x3_2)
        x = self.conv5(x)

        return x


class ReverseAttention(nn.Module):
    """
    反向注意力模块 (Reverse Attention)
    核心思想：使用反向注意力来挖掘被忽略的区域
    """

    def __init__(self, in_channel, out_channel):
        super(ReverseAttention, self).__init__()
        self.convert = nn.Conv2d(in_channel, out_channel, 1)
        self.bn = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(True)

        self.convs = nn.Sequential(
            BasicConv2d(out_channel, out_channel, 3, padding=1),
            BasicConv2d(out_channel, out_channel, 3, padding=1),
            BasicConv2d(out_channel, out_channel, 3, padding=1),
        )
        self.channel = out_channel

    def forward(self, x, map):
        map = F.interpolate(map, size=x.size()[2:], mode='bilinear', align_corners=True)

        # 反向注意力：关注被忽略的区域
        reverse_att = torch.sigmoid(-map)

        x = self.relu(self.bn(self.convert(x)))
        x = x * reverse_att
        x = self.convs(x)

        return x


class Encoder(nn.Module):
    """简化的ResNet风格编码器"""

    def __init__(self, in_channels=3):
        super(Encoder, self).__init__()

        # 初始层
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1)
        )

        # Stage 1
        self.layer1 = self._make_layer(64, 64, 3)
        # Stage 2
        self.layer2 = self._make_layer(64, 128, 4, stride=2)
        # Stage 3
        self.layer3 = self._make_layer(128, 256, 6, stride=2)
        # Stage 4
        self.layer4 = self._make_layer(256, 512, 3, stride=2)

    def _make_layer(self, in_ch, out_ch, blocks, stride=1):
        layers = []
        layers.append(nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        ))
        for _ in range(1, blocks):
            layers.append(nn.Sequential(
                nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            ))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x1 = self.layer1(x)  # 1/4
        x2 = self.layer2(x1)  # 1/8
        x3 = self.layer3(x2)  # 1/16
        x4 = self.layer4(x3)  # 1/32
        return x2, x3, x4


class PraNet(nn.Module):
    """
    PraNet: Parallel Reverse Attention Network

    架构：
    1. 编码器提取多尺度特征
    2. RFB模块增强感受野
    3. 部分解码器聚合特征
    4. 反向注意力模块逐步细化

    参数:
        in_channels: 输入通道数 (RGB=3)
        num_classes: 输出类别数
        channel: 中间通道数
    """

    def __init__(self, in_channels=3, num_classes=1, channel=32):
        super(PraNet, self).__init__()

        # 编码器
        self.encoder = Encoder(in_channels)

        # RFB模块
        self.rfb2 = RFB_modified(128, channel)
        self.rfb3 = RFB_modified(256, channel)
        self.rfb4 = RFB_modified(512, channel)

        # 部分解码器
        self.pd = PartialDecoder(channel)

        # 反向注意力模块
        self.ra4 = ReverseAttention(512, channel)
        self.ra3 = ReverseAttention(256, channel)
        self.ra2 = ReverseAttention(128, channel)

        # 输出层
        self.out4 = nn.Conv2d(channel, num_classes, 1)
        self.out3 = nn.Conv2d(channel, num_classes, 1)
        self.out2 = nn.Conv2d(channel, num_classes, 1)

    def forward(self, x):
        input_size = x.size()[2:]

        # 编码
        x2, x3, x4 = self.encoder(x)

        # RFB增强
        x2_rfb = self.rfb2(x2)
        x3_rfb = self.rfb3(x3)
        x4_rfb = self.rfb4(x4)

        # 部分解码器
        pd_out = self.pd(x4_rfb, x3_rfb, x2_rfb)

        # 反向注意力细化
        ra4_out = self.ra4(x4, pd_out)
        out4 = self.out4(ra4_out)

        ra3_out = self.ra3(x3, out4)
        out3 = self.out3(ra3_out)

        ra2_out = self.ra2(x2, out3)
        out2 = self.out2(ra2_out)

        # 上采样到原始尺寸
        out = F.interpolate(out2, size=input_size, mode='bilinear', align_corners=True)

        return out


# 测试代码
if __name__ == "__main__":
    print("=" * 60)
    print("PraNet 模型测试")
    print("=" * 60)

    model = PraNet(in_channels=3, num_classes=1, channel=32)

    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n参数量: {total_params / 1e6:.2f}M")

    # 测试前向传播
    x = torch.randn(2, 3, 352, 352)
    y = model(x)

    print(f"输入尺寸: {x.shape}")
    print(f"输出尺寸: {y.shape}")

    assert y.shape == (2, 1, 352, 352), "输出尺寸错误！"
    print("\n✅ PraNet 测试通过！")
    print("=" * 60)