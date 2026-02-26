"""
文件名: utils/losses.py
功能: 分割任务损失函数定义
修复: 完全支持混合精度训练（AMP）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """Dice损失函数"""

    def __init__(self, smooth=1e-5):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        # 转成float32计算，避免精度问题
        pred = torch.sigmoid(pred.float())
        target = target.float()

        pred_flat = pred.view(-1)
        target_flat = target.view(-1)

        intersection = (pred_flat * target_flat).sum()
        dice = (2. * intersection + self.smooth) / \
               (pred_flat.sum() + target_flat.sum() + self.smooth)

        return 1 - dice


class BCEDiceLoss(nn.Module):
    """BCE + Dice联合损失"""

    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super(BCEDiceLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, pred, target):
        # 全部转成float32
        pred = pred.float()
        target = target.float()

        bce_loss = self.bce(pred, target)
        dice_loss = self.dice(pred, target)
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


class BoundaryLoss(nn.Module):
    """边界损失函数（完全支持混合精度）"""

    def __init__(self):
        super(BoundaryLoss, self).__init__()

        # Laplacian边缘检测核（不用register_buffer，动态处理）
        self.laplacian_weight = torch.tensor([
            [0, 1, 0],
            [1, -4, 1],
            [0, 1, 0]
        ], dtype=torch.float32).view(1, 1, 3, 3)

    def get_boundary(self, mask):
        """从mask中提取边界"""
        # 确保mask是float32
        mask = mask.float()

        # 把卷积核移到和mask相同的设备，并且类型一致
        laplacian = self.laplacian_weight.to(device=mask.device, dtype=mask.dtype)

        boundary = F.conv2d(mask, laplacian, padding=1)
        boundary = torch.abs(boundary)
        boundary = (boundary > 0.1).float()
        return boundary

    def forward(self, pred_edge, target_mask):
        """
        计算边界损失

        参数:
            pred_edge: 模型预测的边缘 (logits)
            target_mask: 目标分割掩码
        """
        # 全部转成float32
        pred_edge = pred_edge.float()
        target_mask = target_mask.float()

        # 提取边界
        target_boundary = self.get_boundary(target_mask)

        # 使用with_logits版本
        loss = F.binary_cross_entropy_with_logits(pred_edge, target_boundary)

        return loss


class CombinedLoss(nn.Module):
    """组合损失函数"""

    def __init__(self, seg_weight=0.8, boundary_weight=0.2):
        super(CombinedLoss, self).__init__()
        self.seg_loss = BCEDiceLoss()
        self.boundary_loss = BoundaryLoss()
        self.seg_weight = seg_weight
        self.boundary_weight = boundary_weight

    def forward(self, seg_pred, edge_pred, target):
        """
        计算总损失

        参数:
            seg_pred: 分割预测 (logits)
            edge_pred: 边缘预测 (logits)
            target: 目标掩码
        """
        # 全部转成float32，确保混合精度兼容
        seg_pred = seg_pred.float()
        edge_pred = edge_pred.float()
        target = target.float()

        seg_l = self.seg_loss(seg_pred, target)
        boundary_l = self.boundary_loss(edge_pred, target)

        total = self.seg_weight * seg_l + self.boundary_weight * boundary_l

        return total, seg_l, boundary_l