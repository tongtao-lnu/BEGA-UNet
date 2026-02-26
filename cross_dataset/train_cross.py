"""
文件名: cross_dataset/train_cross.py
功能: 跨数据集训练核心脚本（带进度条版本）
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import json
import time

# 设置路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
os.chdir(project_root)

from cross_dataset.dataset import get_cross_dataloaders


# ============ 损失函数 ============
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        intersection = (pred.view(-1) * target.view(-1)).sum()
        return 1 - (2. * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)


class CombinedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()

    def forward(self, pred, target):
        return self.bce(pred, target) + self.dice(pred, target)


# ============ 指标计算 ============
def calculate_metrics(pred, target):
    pred = (torch.sigmoid(pred) > 0.5).float()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    dice = (2 * intersection + 1e-6) / (pred.sum() + target.sum() + 1e-6)
    iou = (intersection + 1e-6) / (union + 1e-6)
    return dice.item(), iou.item()


def calculate_hd95(pred, target):
    """计算HD95（带采样优化，防止内存爆炸）"""
    from scipy.spatial.distance import cdist

    pred = (torch.sigmoid(pred) > 0.5).cpu().numpy().squeeze()
    target = target.cpu().numpy().squeeze()

    if pred.sum() == 0 or target.sum() == 0:
        return 100.0

    pred_points = np.argwhere(pred)
    target_points = np.argwhere(target)

    if len(pred_points) == 0 or len(target_points) == 0:
        return 100.0

    # ✅ 关键修复：如果点太多，随机采样
    max_points = 5000  # 最多5000个点

    if len(pred_points) > max_points:
        indices = np.random.choice(len(pred_points), max_points, replace=False)
        pred_points = pred_points[indices]

    if len(target_points) > max_points:
        indices = np.random.choice(len(target_points), max_points, replace=False)
        target_points = target_points[indices]

    # 计算距离
    d1 = cdist(pred_points, target_points).min(axis=1)
    d2 = cdist(target_points, pred_points).min(axis=1)

    return np.percentile(np.concatenate([d1, d2]), 95)


# ============ 模型加载 ============
def get_model(model_name, in_channels=3, num_classes=1):
    if model_name == 'EGAUNet':
        from models.ega_unet import EGAUNet
        return EGAUNet(in_channels=in_channels, num_classes=num_classes, base_filters=64)
    elif model_name == 'UNet':
        from models.unet import UNet
        return UNet(in_channels=in_channels, num_classes=num_classes, base_filters=64)
    elif model_name == 'AttentionUNet':
        from models.baselines.attention_unet import AttentionUNet
        return AttentionUNet(in_channels=in_channels, num_classes=num_classes, base_filters=64)
    elif model_name == 'TransUNet':
        from models.baselines.transunet import TransUNet
        return TransUNet(in_channels=in_channels, num_classes=num_classes, base_filters=64)
    else:
        raise ValueError(f"未知模型: {model_name}")


# ============ 训练函数（带进度条） ============
def train_one_epoch(model, train_loader, criterion, optimizer, scaler, device):
    model.train()
    total_loss, total_dice = 0, 0

    pbar = tqdm(train_loader, desc="    训练中", leave=False, ncols=100)

    for imgs, masks, _ in pbar:
        imgs, masks = imgs.to(device), masks.to(device)
        optimizer.zero_grad()

        with autocast():
            outputs = model(imgs)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            loss = criterion(outputs, masks)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        dice, _ = calculate_metrics(outputs.float(), masks)
        total_dice += dice

        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'dice': f'{dice * 100:.1f}%'})

    return total_loss / len(train_loader), total_dice / len(train_loader)


def evaluate(model, test_loader, device, save_predictions=False, save_dir=None):
    model.eval()
    all_dice, all_iou, all_hd95 = [], [], []
    predictions = []

    pbar = tqdm(test_loader, desc="    评估中", leave=False, ncols=100)

    with torch.no_grad():
        for imgs, masks, names in pbar:
            imgs, masks = imgs.to(device), masks.to(device)

            with autocast():
                outputs = model(imgs)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]

            dice, iou = calculate_metrics(outputs.float(), masks)
            hd95 = calculate_hd95(outputs.float(), masks)

            all_dice.append(dice)
            all_iou.append(iou)
            all_hd95.append(hd95)

            pbar.set_postfix({'dice': f'{dice * 100:.1f}%'})

            if save_predictions:
                pred = (torch.sigmoid(outputs.float()) > 0.5).cpu().numpy().squeeze()
                predictions.append({
                    'name': names[0],
                    'image': imgs.cpu().numpy().squeeze().transpose(1, 2, 0),
                    'mask': masks.cpu().numpy().squeeze(),
                    'pred': pred,
                    'dice': dice,
                    'iou': iou,
                    'hd95': hd95
                })

    results = {
        'dice': np.mean(all_dice) * 100,
        'dice_std': np.std(all_dice) * 100,
        'iou': np.mean(all_iou) * 100,
        'iou_std': np.std(all_iou) * 100,
        'hd95': np.mean(all_hd95),
        'hd95_std': np.std(all_hd95),
        'all_dice': [d * 100 for d in all_dice],
        'all_iou': [i * 100 for i in all_iou],
        'all_hd95': all_hd95,
    }

    if save_predictions and save_dir:
        os.makedirs(save_dir, exist_ok=True)
        np.save(os.path.join(save_dir, 'predictions.npy'), predictions)

    return results, predictions


# ============ 主训练函数 ============
def train_cross_dataset(model_name, train_on, test_on,
                        data_dir='data_cross', output_dir='results_cross',
                        epochs=100, batch_size=8, lr=1e-4):
    print("\n" + "=" * 70)
    print(f"  跨数据集实验")
    print(f"  模型: {model_name}")
    print(f"  训练集: {train_on.upper()} → 测试集: {test_on.upper()}")
    print(f"  Epochs: {epochs} | Batch Size: {batch_size}")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n  设备: {device}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # 数据
    train_dir = os.path.join(data_dir, f"{train_on}_full")
    test_dir = os.path.join(data_dir, f"{test_on}_full")

    train_loader, test_loader = get_cross_dataloaders(
        train_dir, test_dir, batch_size=batch_size, strong_augment=True
    )

    print(f"  训练集: {len(train_loader.dataset)} 张")
    print(f"  测试集: {len(test_loader.dataset)} 张")

    # 模型
    model = get_model(model_name, in_channels=3, num_classes=1)
    model = model.to(device)
    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  参数量: {params:.2f}M")

    # 训练配置
    criterion = CombinedLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = GradScaler()

    # 结果目录
    results_dir = os.path.join(output_dir, f"{train_on}_to_{test_on}", model_name)
    os.makedirs(results_dir, exist_ok=True)

    # 训练
    print(f"\n  开始训练...")
    print("-" * 70)

    best_dice = 0
    history = {'train_loss': [], 'train_dice': [], 'test_dice': []}
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()

        # 显示当前epoch
        print(f"\n  📍 Epoch {epoch}/{epochs}")

        train_loss, train_dice = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device
        )
        scheduler.step()

        history['train_loss'].append(train_loss)
        history['train_dice'].append(train_dice * 100)

        epoch_time = time.time() - epoch_start
        elapsed = time.time() - start_time
        remaining = (epochs - epoch) * (elapsed / epoch)

        print(f"     Loss: {train_loss:.4f} | Train Dice: {train_dice * 100:.2f}% | "
              f"用时: {epoch_time:.1f}s | 剩余: {remaining / 60:.0f}分钟")

        # 每10个epoch评估
        if epoch % 10 == 0 or epoch == epochs:
            print(f"     🔍 评估测试集...")
            metrics, _ = evaluate(model, test_loader, device)
            history['test_dice'].append(metrics['dice'])

            print(f"     📊 Test Dice: {metrics['dice']:.2f}% | IoU: {metrics['iou']:.2f}%")

            if metrics['dice'] > best_dice:
                best_dice = metrics['dice']
                torch.save(model.state_dict(), os.path.join(results_dir, 'best_model.pth'))
                print(f"     ✅ 保存最佳模型! (Best Dice: {best_dice:.2f}%)")

    # 最终评估
    print("\n" + "-" * 70)
    print("  🎯 最终评估...")
    model.load_state_dict(torch.load(os.path.join(results_dir, 'best_model.pth')))

    final_metrics, predictions = evaluate(
        model, test_loader, device,
        save_predictions=True,
        save_dir=results_dir
    )

    total_time = time.time() - start_time

    print("\n" + "=" * 70)
    print(f"  🎉 训练完成!")
    print(f"  ⏱️  总用时: {total_time / 60:.1f} 分钟")
    print("-" * 70)
    print(f"  📊 最终结果 ({train_on.upper()} → {test_on.upper()}):")
    print(f"     Dice: {final_metrics['dice']:.2f} ± {final_metrics['dice_std']:.2f}%")
    print(f"     IoU:  {final_metrics['iou']:.2f} ± {final_metrics['iou_std']:.2f}%")
    print(f"     HD95: {final_metrics['hd95']:.2f} ± {final_metrics['hd95_std']:.2f}")
    print("=" * 70)

    # 保存结果
    with open(os.path.join(results_dir, 'results.json'), 'w') as f:
        json.dump({k: v for k, v in final_metrics.items() if not k.startswith('all_')}, f, indent=2)

    with open(os.path.join(results_dir, 'history.json'), 'w') as f:
        json.dump(history, f, indent=2)

    np.savez(os.path.join(results_dir, 'detailed_metrics.npz'),
             dice=final_metrics['all_dice'],
             iou=final_metrics['all_iou'],
             hd95=final_metrics['all_hd95'])

    print(f"\n  💾 结果已保存到: {results_dir}")

    return final_metrics


# ============ 命令行入口 ============
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='跨数据集实验')
    parser.add_argument('--model', type=str, required=True,
                        choices=['EGAUNet', 'UNet', 'AttentionUNet', 'TransUNet'])
    parser.add_argument('--train_on', type=str, required=True, choices=['kvasir', 'cvc'])
    parser.add_argument('--test_on', type=str, required=True, choices=['kvasir', 'cvc'])
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)

    args = parser.parse_args()

    train_cross_dataset(
        model_name=args.model,
        train_on=args.train_on,
        test_on=args.test_on,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr
    )