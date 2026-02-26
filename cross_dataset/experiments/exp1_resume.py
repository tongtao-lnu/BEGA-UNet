"""
恢复实验1: 从已保存的模型继续评估
"""
import os
import sys
import json
import numpy as np
import torch
from tqdm import tqdm
from torch.cuda.amp import autocast

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)
os.chdir(project_root)

from cross_dataset.dataset import get_cross_dataloaders
from cross_dataset.train_cross import get_model, calculate_metrics, calculate_hd95


def evaluate_saved_model():
    """评估已保存的模型"""

    print("=" * 70)
    print("  恢复实验 1: EGA-UNet | Kvasir → CVC")
    print("  从已保存的 best_model.pth 继续评估")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 路径
    results_dir = 'results_cross/kvasir_to_cvc/EGAUNet'
    model_path = os.path.join(results_dir, 'best_model.pth')

    if not os.path.exists(model_path):
        print(f"❌ 找不到模型: {model_path}")
        return

    print(f"  ✓ 找到已保存的模型")

    # 加载数据
    _, test_loader = get_cross_dataloaders(
        'data_cross/kvasir_full',
        'data_cross/cvc_full',
        batch_size=8,
        strong_augment=False
    )

    # 加载模型
    model = get_model('EGAUNet', in_channels=3, num_classes=1)
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    model.eval()

    print(f"  ✓ 模型加载成功")
    print(f"  开始最终评估...")

    # 评估
    all_dice, all_iou, all_hd95 = [], [], []
    predictions = []

    with torch.no_grad():
        for imgs, masks, names in tqdm(test_loader, desc="  评估中"):
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

            # 保存预测结果
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

    # 汇总结果
    final_metrics = {
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

    print("\n" + "=" * 70)
    print(f"  🎉 评估完成!")
    print("-" * 70)
    print(f"  📊 最终结果 (Kvasir → CVC):")
    print(f"     Dice: {final_metrics['dice']:.2f} ± {final_metrics['dice_std']:.2f}%")
    print(f"     IoU:  {final_metrics['iou']:.2f} ± {final_metrics['iou_std']:.2f}%")
    print(f"     HD95: {final_metrics['hd95']:.2f} ± {final_metrics['hd95_std']:.2f}")
    print("=" * 70)

    # 保存结果
    with open(os.path.join(results_dir, 'results.json'), 'w') as f:
        json.dump({k: v for k, v in final_metrics.items() if not k.startswith('all_')}, f, indent=2)

    np.savez(os.path.join(results_dir, 'detailed_metrics.npz'),
             dice=final_metrics['all_dice'],
             iou=final_metrics['all_iou'],
             hd95=final_metrics['all_hd95'])

    np.save(os.path.join(results_dir, 'predictions.npy'), predictions)

    print(f"\n  💾 结果已保存到: {results_dir}")
    print("\n  ✅ 实验 1/8 完成！可以继续运行 exp2_EGAUNet_C2K.py")

    return final_metrics


if __name__ == "__main__":
    evaluate_saved_model()