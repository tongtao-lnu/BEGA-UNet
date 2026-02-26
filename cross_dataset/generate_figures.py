"""
生成论文图表
在所有实验完成后运行此脚本
直接右键运行
"""
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
os.chdir(project_root)

from cross_dataset.visualize import JournalVisualizer

if __name__ == "__main__":
    print("=" * 60)
    print("  生成论文图表")
    print("=" * 60)

    results_dir = 'results_cross'

    # 检查结果
    required = [
        'kvasir_to_cvc/EGAUNet/results.json',
        'kvasir_to_cvc/UNet/results.json',
        'cvc_to_kvasir/EGAUNet/results.json',
        'cvc_to_kvasir/UNet/results.json',
    ]

    missing = []
    for r in required:
        if not os.path.exists(os.path.join(results_dir, r)):
            missing.append(r)

    if missing:
        print("\n⚠️ 警告: 以下实验结果缺失:")
        for m in missing:
            print(f"   - {m}")
        print("\n仍将尝试生成已有结果的图表...\n")

    # 生成图表
    viz = JournalVisualizer(results_dir)
    viz.generate_all_figures()

    print("\n" + "=" * 60)
    print("  ✅ 图表生成完成!")
    print("  输出目录: results_cross/figures/")
    print("=" * 60)