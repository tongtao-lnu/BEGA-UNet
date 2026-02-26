"""
文件名: train.py
功能: EGA-UNet训练主脚本（息肉分割版本）
"""

import os
import time
import numpy as np
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

from models.ega_unet import EGAUNet
from utils.dataset import get_dataloaders
from utils.losses import CombinedLoss
from utils.metrics import SegmentationMetrics


def set_seed(seed=42):
    """设置随机种子"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def train_epoch(model, loader, criterion, optimizer, scaler, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    metrics = SegmentationMetrics()

    pbar = tqdm(loader, desc='训练中', leave=False)
    for images, masks in pbar:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast():
            seg_out, edge_out, _ = model(images)
            loss, seg_loss, boundary_loss = criterion(seg_out, edge_out, masks)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

        with torch.no_grad():
            pred = torch.sigmoid(seg_out.float())
            metrics.update(pred, masks)

        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    return total_loss / len(loader), metrics.get_metrics()


def validate(model, loader, criterion, device):
    """验证"""
    model.eval()
    total_loss = 0
    metrics = SegmentationMetrics()

    with torch.no_grad():
        for images, masks in tqdm(loader, desc='验证中', leave=False):
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            with autocast():
                seg_out, edge_out, _ = model(images)
                loss, _, _ = criterion(seg_out, edge_out, masks)

            total_loss += loss.item()
            pred = torch.sigmoid(seg_out.float())
            metrics.update(pred, masks)

    return total_loss / len(loader), metrics.get_metrics()


def main():
    # ==================== 配置参数 ====================
    data_dir = './processed_data'
    output_dir = './outputs'

    epochs = 100
    batch_size = 8  # 352x352图像，8GB显存用8
    lr = 1e-4
    seed = 42
    num_workers = 0  # Windows必须为0
    val_interval = 3  # 每3个epoch验证一次
    # =================================================

    set_seed(seed)

    # 创建输出目录
    ckpt_dir = os.path.join(output_dir, 'checkpoints', 'ega_unet')
    log_dir = os.path.join(output_dir, 'logs', 'ega_unet')
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("=" * 60)
    print("EGA-UNet 息肉分割训练")
    print("=" * 60)
    print(f"设备: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f} GB")

    # 加载数据
    print("\n加载数据...")
    train_loader, val_loader, test_loader = get_dataloaders(
        data_dir, batch_size, num_workers
    )
    print(f"训练集: {len(train_loader.dataset)} 张")
    print(f"验证集: {len(val_loader.dataset)} 张")
    print(f"测试集: {len(test_loader.dataset)} 张")

    # 创建模型（注意：in_channels=3，因为是RGB图像）
    print("\n创建模型...")
    model = EGAUNet(in_channels=3, num_classes=1, base_filters=64)
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params / 1e6:.2f}M")

    # 损失函数和优化器
    criterion = CombinedLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = GradScaler()

    # TensorBoard
    writer = SummaryWriter(log_dir)

    # 训练
    best_dice = 0
    best_epoch = 0
    best_metrics = {}

    print("\n" + "=" * 60)
    print("开始训练")
    print(f"总轮数: {epochs}, Batch大小: {batch_size}")
    print("=" * 60)

    start_time = time.time()

    for epoch in range(epochs):
        epoch_start = time.time()
        current_lr = optimizer.param_groups[0]['lr']

        # 训练
        train_loss, train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, scaler, device
        )
        scheduler.step()

        epoch_time = time.time() - epoch_start

        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Dice/train', train_metrics['Dice'], epoch)
        writer.add_scalar('LR', current_lr, epoch)

        # 验证
        if (epoch + 1) % val_interval == 0 or epoch == epochs - 1:
            val_loss, val_metrics = validate(model, val_loader, criterion, device)

            writer.add_scalar('Loss/val', val_loss, epoch)
            writer.add_scalar('Dice/val', val_metrics['Dice'], epoch)
            writer.add_scalar('IoU/val', val_metrics['IoU'], epoch)

            print(f"\n[Epoch {epoch + 1:3d}/{epochs}] ({epoch_time:.1f}s) LR={current_lr:.2e}")
            print(f"  训练 │ Loss: {train_loss:.4f} │ Dice: {train_metrics['Dice']:.4f}")
            print(f"  验证 │ Loss: {val_loss:.4f} │ Dice: {val_metrics['Dice']:.4f} │ IoU: {val_metrics['IoU']:.4f}")

            if val_metrics['Dice'] > best_dice:
                best_dice = val_metrics['Dice']
                best_epoch = epoch + 1
                best_metrics = val_metrics.copy()

                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_dice': best_dice,
                    'val_metrics': val_metrics,
                }, os.path.join(ckpt_dir, 'best_model.pth'))

                print(f"  ✅ 保存最佳模型! Dice={best_dice:.4f}")
        else:
            if (epoch + 1) % 10 == 0:
                print(f"[Epoch {epoch + 1:3d}/{epochs}] Loss: {train_loss:.4f} │ Dice: {train_metrics['Dice']:.4f}")

        if (epoch + 1) % 10 == 0:
            torch.cuda.empty_cache()

    total_time = time.time() - start_time
    writer.close()

    print("\n" + "=" * 60)
    print("🎉 训练完成!")
    print("=" * 60)
    print(f"总用时: {total_time / 60:.1f} 分钟")
    print(f"最佳Epoch: {best_epoch}")
    print(f"最佳验证Dice: {best_dice:.4f}")
    print(f"模型保存位置: {os.path.join(ckpt_dir, 'best_model.pth')}")


if __name__ == '__main__':
    main()