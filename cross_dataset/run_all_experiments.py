"""
文件名: cross_dataset/run_all_experiments.py
功能: 生成论文图表
"""

import os
import sys
import json
import pandas as pd
from datetime import datetime

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
os.chdir(project_root)


class Args:
    """参数配置"""

    def __init__(self):
        self.data_dir = 'data_cross'
        self.output_dir = 'results_cross'
        self.epochs = 100
        self.batch_size = 8
        self.lr = 1e-4
        self.strong_augment = True


def run_all():
    """运行所有实验"""
    from cross_dataset.train_cross import train_cross_dataset
    from cross_dataset.visualize import JournalVisualizer

    # 按导师要求：4个模型
    models = ['EGAUNet', 'UNet', 'AttentionUNet', 'TransUNet']

    # 两个方向
    experiments = [
        ('kvasir', 'cvc'),
        ('cvc', 'kvasir'),
    ]

    all_results = []

    print("=" * 80)
    print("  跨数据集实验 - Cross-Dataset Evaluation")
    print("=" * 80)
    print(f"  模型: {models}")
    print(f"  实验方向: Kvasir↔CVC")
    print(f"  Epochs: 100")
    print("=" * 80)

    # ============ 运行所有实验 ============
    for train_on, test_on in experiments:
        for model_name in models:
            print(f"\n{'=' * 80}")
            print(f"  实验: {model_name}")
            print(f"  训练: {train_on.upper()} → 测试: {test_on.upper()}")
            print(f"{'=' * 80}")

            args = Args()
            args.model = model_name
            args.train_on = train_on
            args.test_on = test_on

            if model_name == 'TransUNet':
                args.batch_size = 4

            try:
                metrics, _ = train_cross_dataset(args)

                result = {
                    'Model': model_name,
                    'Train': train_on.upper(),
                    'Test': test_on.upper(),
                    'Dice': metrics['dice'],
                    'Dice_std': metrics['dice_std'],
                    'IoU': metrics['iou'],
                    'IoU_std': metrics['iou_std'],
                    'HD95': metrics['hd95'],
                    'HD95_std': metrics['hd95_std'],
                }
                all_results.append(result)

            except Exception as e:
                print(f"❌ 实验失败: {e}")
                import traceback
                traceback.print_exc()

    # ============ 汇总结果 ============
    print("\n" + "=" * 80)
    print("  实验结果汇总")
    print("=" * 80)

    df = pd.DataFrame(all_results)

    # 格式化显示
    for direction in [('KVASIR', 'CVC'), ('CVC', 'KVASIR')]:
        print(f"\n  {direction[0]} → {direction[1]}:")
        subset = df[(df['Train'] == direction[0]) & (df['Test'] == direction[1])]
        if len(subset) > 0:
            subset = subset.sort_values('Dice', ascending=False)
            for _, row in subset.iterrows():
                print(f"    {row['Model']:15s}: Dice={row['Dice']:.2f}±{row['Dice_std']:.2f}%  "
                      f"IoU={row['IoU']:.2f}%  HD95={row['HD95']:.2f}")

    # 保存CSV
    output_dir = 'results_cross'
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(output_dir, f'cross_dataset_results_{timestamp}.csv')
    df.to_csv(csv_path, index=False)
    print(f"\n  结果已保存: {csv_path}")

    # ============ 生成论文图表 ============
    print("\n" + "=" * 80)
    print("  生成论文图表")
    print("=" * 80)

    viz = JournalVisualizer(output_dir)
    viz.generate_all_figures()

    print("\n" + "=" * 80)
    print("  ✅ 所有实验完成!")
    print("=" * 80)
    print(f"  结果目录: {output_dir}/")
    print(f"  图表目录: {output_dir}/figures/")
    print("=" * 80)


if __name__ == "__main__":
    run_all()
