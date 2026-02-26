"""
文件名: utils/metrics.py
功能: 分割评价指标计算
"""

import numpy as np
import torch
from scipy.ndimage import distance_transform_edt, binary_erosion


class SegmentationMetrics:
    """分割评价指标计算器"""

    def __init__(self, threshold=0.5):
        self.threshold = threshold
        self.reset()

    def reset(self):
        self.iou_list = []
        self.dice_list = []
        self.precision_list = []
        self.recall_list = []
        self.hd95_list = []

    def update(self, pred, target):
        if isinstance(pred, torch.Tensor):
            pred = pred.detach().cpu().numpy()
        if isinstance(target, torch.Tensor):
            target = target.detach().cpu().numpy()

        pred = (pred > self.threshold).astype(np.float32)
        target = (target > self.threshold).astype(np.float32)

        batch_size = pred.shape[0]

        for i in range(batch_size):
            p = pred[i].flatten()
            t = target[i].flatten()

            intersection = (p * t).sum()
            union = (p + t - p * t).sum()

            iou = (intersection + 1e-8) / (union + 1e-8)
            self.iou_list.append(iou)

            dice = (2 * intersection + 1e-8) / (p.sum() + t.sum() + 1e-8)
            self.dice_list.append(dice)

            precision = (intersection + 1e-8) / (p.sum() + 1e-8)
            self.precision_list.append(precision)

            recall = (intersection + 1e-8) / (t.sum() + 1e-8)
            self.recall_list.append(recall)

            pred_2d = pred[i].squeeze()
            target_2d = target[i].squeeze()

            if pred_2d.sum() > 10 and target_2d.sum() > 10:
                hd95 = self._compute_hd95(pred_2d, target_2d)
                if hd95 is not None:
                    self.hd95_list.append(hd95)

    def _compute_hd95(self, pred, target):
        try:
            pred_boundary = self._get_boundary(pred)
            target_boundary = self._get_boundary(target)

            if pred_boundary.sum() == 0 or target_boundary.sum() == 0:
                return None

            pred_dist = distance_transform_edt(~pred.astype(bool))
            target_dist = distance_transform_edt(~target.astype(bool))

            dist_pred_to_target = pred_dist[target_boundary > 0]
            dist_target_to_pred = target_dist[pred_boundary > 0]

            all_distances = np.concatenate([dist_pred_to_target, dist_target_to_pred])
            hd95 = np.percentile(all_distances, 95)

            return hd95

        except Exception:
            return None

    def _get_boundary(self, mask):
        eroded = binary_erosion(mask.astype(bool), iterations=1)
        boundary = mask.astype(float) - eroded.astype(float)
        return boundary

    def get_metrics(self):
        return {
            'IoU': np.mean(self.iou_list) if self.iou_list else 0,
            'Dice': np.mean(self.dice_list) if self.dice_list else 0,
            'Precision': np.mean(self.precision_list) if self.precision_list else 0,
            'Recall': np.mean(self.recall_list) if self.recall_list else 0,
            'HD95': np.mean(self.hd95_list) if self.hd95_list else 0,
        }


def print_metrics(metrics, prefix=''):
    print(f"{prefix}IoU: {metrics['IoU']:.4f} | "
          f"Dice: {metrics['Dice']:.4f} | "
          f"Precision: {metrics['Precision']:.4f} | "
          f"Recall: {metrics['Recall']:.4f} | "
          f"HD95: {metrics['HD95']:.2f}")