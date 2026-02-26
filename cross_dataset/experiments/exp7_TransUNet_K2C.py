"""
实验 7/8: TransUNet (Kvasir → CVC)
预计时间: 40-50分钟
⚠️ 显存占用较大，使用batch_size=4
直接右键运行此文件
"""
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)
os.chdir(project_root)

from cross_dataset.train_cross import train_cross_dataset

if __name__ == "__main__":
    print("=" * 70)
    print("  实验 7/8: TransUNet | Kvasir-SEG → CVC-ClinicDB")
    print("  ⚠️ 显存占用较大，使用batch_size=4")
    print("=" * 70)

    train_cross_dataset(
        model_name='TransUNet',
        train_on='kvasir',
        test_on='cvc',
        epochs=100,
        batch_size=4,  # TransUNet显存大，用4
        lr=1e-4
    )