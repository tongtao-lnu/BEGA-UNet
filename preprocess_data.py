"""
文件名: preprocess_data.py
功能: 息肉分割数据集预处理（Kvasir-SEG + CVC-ClinicDB）
"""

import os
import numpy as np
import cv2
from glob import glob
from tqdm import tqdm
from sklearn.model_selection import train_test_split


class PolypPreprocessor:
    """息肉分割数据预处理器"""

    def __init__(self, data_dir="./data", output_dir="./processed_data",
                 img_size=(352, 352)):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.img_size = img_size

        # 创建输出目录
        for split in ['train', 'val', 'test']:
            os.makedirs(os.path.join(output_dir, split, 'images'), exist_ok=True)
            os.makedirs(os.path.join(output_dir, split, 'masks'), exist_ok=True)

    def find_all_images(self):
        """查找所有图像文件"""
        img_dir = os.path.join(self.data_dir, "images")

        # 支持多种图像格式
        patterns = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif',
                    '*.JPG', '*.JPEG', '*.PNG', '*.BMP', '*.TIF']

        img_files = []
        for pattern in patterns:
            img_files.extend(glob(os.path.join(img_dir, pattern)))

        # 去重（以防大小写重复）
        img_files = list(set(img_files))

        print(f"找到 {len(img_files)} 个图像文件")
        return sorted(img_files)

    def find_matching_mask(self, img_path):
        """为图像找到匹配的掩码"""
        mask_dir = os.path.join(self.data_dir, "masks")

        # 获取文件名（不含路径和扩展名）
        img_basename = os.path.basename(img_path)
        img_name = os.path.splitext(img_basename)[0]

        # 尝试各种可能的掩码文件名
        possible_names = [
            img_name,  # 完全相同的名字
            img_name + '_mask',  # 添加_mask后缀
            img_name.replace('image', 'mask'),  # image替换为mask
        ]

        possible_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tif',
                               '.PNG', '.JPG', '.JPEG', '.BMP', '.TIF']

        for name in possible_names:
            for ext in possible_extensions:
                mask_path = os.path.join(mask_dir, name + ext)
                if os.path.exists(mask_path):
                    return mask_path

        return None

    def load_and_process_image(self, img_path):
        """加载并处理图像"""
        # 读取图像（OpenCV默认BGR格式）
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            return None

        # BGR转RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 调整尺寸
        img = cv2.resize(img, self.img_size, interpolation=cv2.INTER_LINEAR)

        # 归一化到[0, 1]
        img = img.astype(np.float32) / 255.0

        return img

    def load_and_process_mask(self, mask_path):
        """加载并处理掩码"""
        # 读取掩码（灰度图）
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            # 尝试以彩色读取然后转灰度
            mask = cv2.imread(mask_path, cv2.IMREAD_COLOR)
            if mask is not None:
                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            else:
                return None

        # 调整尺寸
        mask = cv2.resize(mask, self.img_size, interpolation=cv2.INTER_NEAREST)

        # 二值化（大于127的为1，否则为0）
        mask = (mask > 127).astype(np.float32)

        return mask

    def run(self, train_ratio=0.7, val_ratio=0.15):
        """执行完整预处理流程"""
        print("=" * 60)
        print("息肉分割数据预处理")
        print("=" * 60)

        # 找到所有图像
        img_files = self.find_all_images()

        if len(img_files) == 0:
            print("❌ 错误：未找到任何图像文件！")
            print(f"   请检查文件夹：{os.path.join(self.data_dir, 'images')}")
            return

        # 匹配图像和掩码
        print("\n匹配图像和掩码...")
        valid_pairs = []

        for img_path in tqdm(img_files, desc="匹配中"):
            mask_path = self.find_matching_mask(img_path)
            if mask_path is not None:
                valid_pairs.append((img_path, mask_path))

        print(f"成功匹配 {len(valid_pairs)} 对图像-掩码")

        if len(valid_pairs) == 0:
            print("❌ 错误：没有找到匹配的图像-掩码对！")
            print("   请检查masks文件夹中的文件名是否与images对应")
            return

        # 数据集划分
        print("\n划分数据集...")
        np.random.seed(42)

        indices = np.arange(len(valid_pairs))

        # 计算各集合大小
        test_ratio = 1 - train_ratio - val_ratio

        # 先分出测试集
        train_val_idx, test_idx = train_test_split(
            indices, test_size=test_ratio, random_state=42
        )

        # 再从训练验证集中分出验证集
        relative_val_ratio = val_ratio / (train_ratio + val_ratio)
        train_idx, val_idx = train_test_split(
            train_val_idx, test_size=relative_val_ratio, random_state=42
        )

        split_indices = {
            'train': train_idx,
            'val': val_idx,
            'test': test_idx
        }

        print(f"  训练集: {len(train_idx)} 对")
        print(f"  验证集: {len(val_idx)} 对")
        print(f"  测试集: {len(test_idx)} 对")

        # 处理并保存
        saved_counts = {'train': 0, 'val': 0, 'test': 0}

        for split, indices in split_indices.items():
            print(f"\n处理 {split} 集...")

            img_out_dir = os.path.join(self.output_dir, split, 'images')
            mask_out_dir = os.path.join(self.output_dir, split, 'masks')

            for idx in tqdm(indices, desc=f"{split}"):
                img_path, mask_path = valid_pairs[idx]

                # 加载并处理
                img = self.load_and_process_image(img_path)
                mask = self.load_and_process_mask(mask_path)

                if img is None or mask is None:
                    continue

                # 跳过没有息肉的图像（掩码几乎全黑）
                if mask.sum() < 100:
                    continue

                # 生成保存文件名
                basename = os.path.splitext(os.path.basename(img_path))[0]
                save_name = f"{basename}.npy"

                # 保存
                np.save(os.path.join(img_out_dir, save_name), img)
                np.save(os.path.join(mask_out_dir, save_name), mask)

                saved_counts[split] += 1

        # 打印结果
        print("\n" + "=" * 60)
        print("✅ 数据预处理完成！")
        print("=" * 60)
        print(f"训练集: {saved_counts['train']} 张")
        print(f"验证集: {saved_counts['val']} 张")
        print(f"测试集: {saved_counts['test']} 张")
        print(f"图像尺寸: {self.img_size}")
        print(f"图像格式: RGB, 归一化到[0,1]")
        print(f"掩码格式: 二值化(0/1)")
        print(f"输出目录: {os.path.abspath(self.output_dir)}")
        print("=" * 60)


if __name__ == "__main__":
    preprocessor = PolypPreprocessor(
        data_dir="./data",
        output_dir="./processed_data",
        img_size=(352, 352)
    )
    preprocessor.run(train_ratio=0.7, val_ratio=0.15)