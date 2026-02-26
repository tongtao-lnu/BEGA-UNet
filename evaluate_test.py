"""
文件名: evaluate_test.py
功能: EGA-UNet测试集评估 + 论文图表生成
"""

import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Agg')
import pandas as pd
from datetime import datetime
import json
import warnings

warnings.filterwarnings('ignore')

from models.ega_unet import EGAUNet
from utils.dataset import MedicalDataset
from utils.metrics import SegmentationMetrics

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300


class TestEvaluator:
    """测试集评估器"""

    def __init__(self, model_path, data_dir, output_dir, device=None):
        self.model_path = model_path
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.results_dir = os.path.join(output_dir, 'test_results')
        self.figures_dir = os.path.join(self.results_dir, 'figures')
        self.vis_dir = os.path.join(self.results_dir, 'visualizations')

        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.figures_dir, exist_ok=True)
        os.makedirs(self.vis_dir, exist_ok=True)

        self.sample_metrics = []

    def load_model(self):
        """加载模型"""
        print("=" * 60)
        print("加载模型...")

        # 创建模型（in_channels=3用于RGB）
        self.model = EGAUNet(in_channels=3, num_classes=1, base_filters=64)

        checkpoint = torch.load(self.model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()

        params = sum(p.numel() for p in self.model.parameters()) / 1e6
        print(f"✅ 模型加载成功")
        print(f"   参数量: {params:.2f}M")
        print(f"   训练最佳Dice: {checkpoint.get('best_dice', 'N/A')}")

    def load_test_data(self):
        """加载测试数据"""
        print("\n加载测试数据...")

        self.test_dataset = MedicalDataset(self.data_dir, split='test', augment=False)
        self.test_loader = DataLoader(
            self.test_dataset, batch_size=1, shuffle=False,
            num_workers=0, pin_memory=True
        )

        print(f"✅ 测试集: {len(self.test_dataset)} 张")

    def compute_metrics(self, pred, target, threshold=0.5):
        """计算指标"""
        pred_bin = (pred > threshold).astype(np.float32)
        target_bin = (target > threshold).astype(np.float32)

        pred_flat = pred_bin.flatten()
        target_flat = target_bin.flatten()

        tp = (pred_flat * target_flat).sum()
        fp = (pred_flat * (1 - target_flat)).sum()
        fn = ((1 - pred_flat) * target_flat).sum()
        tn = ((1 - pred_flat) * (1 - target_flat)).sum()

        eps = 1e-8
        dice = (2 * tp + eps) / (2 * tp + fp + fn + eps)
        iou = (tp + eps) / (tp + fp + fn + eps)
        precision = (tp + eps) / (tp + fp + eps)
        recall = (tp + eps) / (tp + fn + eps)

        # HD95
        hd95 = self._compute_hd95(pred_bin.squeeze(), target_bin.squeeze())

        return {
            'Dice': float(dice),
            'IoU': float(iou),
            'Precision': float(precision),
            'Recall': float(recall),
            'HD95': float(hd95) if hd95 else np.nan
        }

    def _compute_hd95(self, pred, target):
        """计算HD95"""
        try:
            from scipy.ndimage import distance_transform_edt, binary_erosion

            if pred.sum() < 10 or target.sum() < 10:
                return None

            pred_boundary = pred - binary_erosion(pred).astype(float)
            target_boundary = target - binary_erosion(target).astype(float)

            if pred_boundary.sum() == 0 or target_boundary.sum() == 0:
                return None

            pred_dist = distance_transform_edt(~pred.astype(bool))
            target_dist = distance_transform_edt(~target.astype(bool))

            d1 = pred_dist[target_boundary > 0]
            d2 = target_dist[pred_boundary > 0]

            if len(d1) == 0 or len(d2) == 0:
                return None

            all_dist = np.concatenate([d1, d2])
            return np.percentile(all_dist, 95)
        except:
            return None

    def evaluate(self):
        """评估"""
        print("\n" + "=" * 60)
        print("开始测试集评估")
        print("=" * 60)

        self.sample_metrics = []

        with torch.no_grad():
            for idx, (images, masks) in enumerate(tqdm(self.test_loader, desc='测试中')):
                images = images.to(self.device)
                masks = masks.to(self.device)

                with autocast():
                    seg_out, _, _ = self.model(images)

                pred = torch.sigmoid(seg_out).cpu().numpy().squeeze()
                mask = masks.cpu().numpy().squeeze()
                img = images.cpu().numpy().squeeze().transpose(1, 2, 0)  # (3,H,W) -> (H,W,3)

                metrics = self.compute_metrics(pred, mask)
                metrics['sample_idx'] = idx
                metrics['filename'] = self.test_dataset.files[idx]
                self.sample_metrics.append(metrics)

                # 保存可视化
                if idx % 10 == 0 or idx < 5:
                    self._save_vis(img, mask, pred, idx, metrics)

        self._compute_overall()
        self._save_results()

        return self.overall_metrics

    def _save_vis(self, img, mask, pred, idx, metrics):
        """保存可视化"""
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))

        # RGB图像直接显示
        axes[0].imshow(img)
        axes[0].set_title('Input Image')
        axes[0].axis('off')

        axes[1].imshow(img)
        axes[1].imshow(mask, cmap='Reds', alpha=0.5)
        axes[1].set_title('Ground Truth')
        axes[1].axis('off')

        axes[2].imshow(img)
        axes[2].imshow(pred > 0.5, cmap='Blues', alpha=0.5)
        axes[2].set_title('Prediction')
        axes[2].axis('off')

        # 对比图
        comp = np.zeros((*img.shape[:2], 3))
        tp = ((pred > 0.5) & (mask > 0.5))
        fn = ((pred <= 0.5) & (mask > 0.5))
        fp = ((pred > 0.5) & (mask <= 0.5))
        comp[tp] = [0, 1, 0]  # 绿色
        comp[fn] = [1, 0, 0]  # 红色
        comp[fp] = [0, 0, 1]  # 蓝色

        axes[3].imshow(img)
        axes[3].imshow(comp, alpha=0.5)
        axes[3].set_title(f'Dice={metrics["Dice"]:.4f}')
        axes[3].axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(self.vis_dir, f'sample_{idx:04d}.png'), dpi=150)
        plt.close()

    def _compute_overall(self):
        """计算整体指标"""
        df = pd.DataFrame(self.sample_metrics)

        self.overall_metrics = {}
        for col in ['Dice', 'IoU', 'Precision', 'Recall', 'HD95']:
            values = df[col].dropna().values
            if len(values) > 0:
                self.overall_metrics[col] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'median': float(np.median(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values))
                }

    def _save_results(self):
        """保存结果"""
        df = pd.DataFrame(self.sample_metrics)
        df.to_csv(os.path.join(self.results_dir, 'sample_metrics.csv'), index=False)

        with open(os.path.join(self.results_dir, 'overall_metrics.json'), 'w') as f:
            json.dump(self.overall_metrics, f, indent=2)

        print(f"\n✅ 结果已保存到: {self.results_dir}")

    def generate_figures(self):
        """生成图表"""
        print("\n生成论文图表...")

        df = pd.DataFrame(self.sample_metrics)

        # 箱线图
        fig, ax = plt.subplots(figsize=(10, 6))
        metrics = ['Dice', 'IoU', 'Precision', 'Recall']
        data = [df[m].dropna().values for m in metrics]
        bp = ax.boxplot(data, labels=metrics, patch_artist=True)
        colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax.set_ylabel('Score')
        ax.set_title('Test Set Performance')
        ax.grid(True, alpha=0.3)
        plt.savefig(os.path.join(self.figures_dir, 'boxplot.png'), dpi=300, bbox_inches='tight')
        plt.close()

        print("  ✅ 图表已生成")

    def print_summary(self):
        """打印摘要"""
        print("\n" + "=" * 60)
        print("📊 测试集评估结果")
        print("=" * 60)

        print(f"\n测试样本数: {len(self.sample_metrics)}")
        print("\n核心指标:")
        print("-" * 40)

        for metric in ['Dice', 'IoU', 'Precision', 'Recall', 'HD95']:
            if metric in self.overall_metrics:
                v = self.overall_metrics[metric]
                print(f"  {metric:<12}: {v['mean']:.4f} ± {v['std']:.4f}")


def main():
    print("=" * 60)
    print("EGA-UNet 测试集评估")
    print("=" * 60)

    model_path = './outputs/checkpoints/ega_unet/best_model.pth'
    data_dir = './processed_data'
    output_dir = './results'

    if not os.path.exists(model_path):
        print(f"❌ 模型不存在: {model_path}")
        print("请先完成训练！")
        return

    evaluator = TestEvaluator(model_path, data_dir, output_dir)
    evaluator.load_model()
    evaluator.load_test_data()
    evaluator.evaluate()
    evaluator.generate_figures()
    evaluator.print_summary()

    print("\n" + "=" * 60)
    print("🎉 评估完成!")
    print(f"📁 结果: {os.path.abspath(evaluator.results_dir)}")
    print("=" * 60)


if __name__ == '__main__':
    main()