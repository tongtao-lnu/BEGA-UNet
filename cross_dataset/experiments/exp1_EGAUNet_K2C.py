"""
实验 1/8: EGA-UNet (Kvasir → CVC)
预计时间: 30-40分钟
直接右键运行此文件
"""
import os
import sys

# 设置路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)
os.chdir(project_root)

from cross_dataset.train_cross import train_cross_dataset

if __name__ == "__main__":
    print("=" * 70)
    print("  实验 1/8: EGA-UNet | Kvasir-SEG → CVC-ClinicDB")
    print("=" * 70)

    train_cross_dataset(
        model_name='EGAUNet',
        train_on='kvasir',
        test_on='cvc',
        epochs=100,
        batch_size=8,
        lr=1e-4
    )