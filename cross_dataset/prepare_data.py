"""
文件名: cross_dataset/prepare_data.py
功能: 准备跨数据集实验数据（RGB彩色，统一预处理）
修复: 自动检测项目根目录，添加路径诊断
"""

import os
import sys
import cv2
import numpy as np
from tqdm import tqdm


def find_project_root():
    """自动查找项目根目录"""
    # 当前脚本所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # 向上查找，直到找到 data_raw 目录
    search_dir = current_dir
    for _ in range(5):  # 最多向上5层
        if os.path.exists(os.path.join(search_dir, 'data_raw')):
            return search_dir
        parent = os.path.dirname(search_dir)
        if parent == search_dir:  # 到达根目录
            break
        search_dir = parent

    # 如果找不到，使用脚本上一级目录
    return os.path.dirname(current_dir)


def check_data_structure(raw_dir):
    """检查并诊断数据目录结构"""
    print("\n[诊断] 检查数据目录结构...")
    print(f"  原始数据目录: {os.path.abspath(raw_dir)}")

    if not os.path.exists(raw_dir):
        print(f"  ❌ 目录不存在: {raw_dir}")
        return False, None, None

    # 列出 data_raw 下的内容
    print(f"  目录内容:")
    for item in os.listdir(raw_dir):
        item_path = os.path.join(raw_dir, item)
        if os.path.isdir(item_path):
            sub_items = os.listdir(item_path)[:5]  # 只显示前5个
            print(f"    📁 {item}/")
            for sub in sub_items:
                print(f"        └── {sub}")
            if len(os.listdir(item_path)) > 5:
                print(f"        └── ... (共{len(os.listdir(item_path))}项)")

    # 检测 Kvasir 路径
    kvasir_candidates = [
        'Kvasir-SEG',
        'kvasir-seg',
        'Kvasir_SEG',
        'kvasir',
        'Kvasir'
    ]

    kvasir_path = None
    for candidate in kvasir_candidates:
        test_path = os.path.join(raw_dir, candidate)
        if os.path.exists(test_path):
            kvasir_path = test_path
            break

    # 检测 CVC 路径
    cvc_candidates = [
        'CVC-ClinicDB',
        'CVC_ClinicDB',
        'cvc-clinicdb',
        'CVC',
        'cvc'
    ]

    cvc_path = None
    for candidate in cvc_candidates:
        test_path = os.path.join(raw_dir, candidate)
        if os.path.exists(test_path):
            cvc_path = test_path
            break

    print(f"\n  检测结果:")
    print(f"    Kvasir路径: {kvasir_path if kvasir_path else '❌ 未找到'}")
    print(f"    CVC路径: {cvc_path if cvc_path else '❌ 未找到'}")

    return True, kvasir_path, cvc_path


def find_image_mask_dirs(dataset_path, dataset_name):
    """自动检测图像和mask目录"""
    if dataset_path is None:
        return None, None

    # Kvasir 可能的目录结构
    if 'kvasir' in dataset_name.lower():
        img_candidates = ['images', 'image', 'Images', 'imgs']
        mask_candidates = ['masks', 'mask', 'Masks', 'groundtruth', 'gt']
    else:  # CVC
        img_candidates = ['Original', 'original', 'images', 'Images', 'image']
        mask_candidates = ['Ground Truth', 'GroundTruth', 'groundtruth', 'masks', 'Masks', 'GT', 'gt']

    img_dir = None
    mask_dir = None

    # 检查直接子目录
    for candidate in img_candidates:
        test_path = os.path.join(dataset_path, candidate)
        if os.path.exists(test_path):
            img_dir = test_path
            break

    for candidate in mask_candidates:
        test_path = os.path.join(dataset_path, candidate)
        if os.path.exists(test_path):
            mask_dir = test_path
            break

    # 如果找不到，可能图像直接在根目录
    if img_dir is None:
        files = os.listdir(dataset_path)
        img_files = [f for f in files if f.lower().endswith(('.jpg', '.png', '.jpeg', '.tif', '.bmp'))]
        if len(img_files) > 10:  # 如果有很多图像文件
            img_dir = dataset_path

    return img_dir, mask_dir


def prepare_kvasir(raw_dir, output_dir, img_size=352):
    """准备Kvasir-SEG数据集"""

    # 自动检测目录
    img_dir, mask_dir = find_image_mask_dirs(raw_dir, 'kvasir')

    if img_dir is None or mask_dir is None:
        print(f"[Kvasir] ❌ 无法找到图像或mask目录")
        print(f"  尝试的路径: {raw_dir}")
        print(f"  检测到的图像目录: {img_dir}")
        print(f"  检测到的mask目录: {mask_dir}")
        return 0

    print(f"[Kvasir] 图像目录: {img_dir}")
    print(f"[Kvasir] Mask目录: {mask_dir}")

    out_img_dir = os.path.join(output_dir, "images")
    out_mask_dir = os.path.join(output_dir, "masks")
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_mask_dir, exist_ok=True)

    files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    print(f"[Kvasir] 找到 {len(files)} 张图像")

    if len(files) == 0:
        print(f"  ❌ 没有找到图像文件！请检查目录: {img_dir}")
        return 0

    count = 0
    for fname in tqdm(files, desc="[Kvasir] 处理中"):
        try:
            # 读取图像 (BGR -> RGB)
            img_path = os.path.join(img_dir, fname)
            img = cv2.imread(img_path)
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # 尝试多种mask文件名
            base_name = os.path.splitext(fname)[0]
            mask = None

            for ext in ['.png', '.jpg', '.jpeg', '.bmp']:
                mask_path = os.path.join(mask_dir, base_name + ext)
                if os.path.exists(mask_path):
                    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                    break

            # 也尝试原始文件名
            if mask is None:
                mask_path = os.path.join(mask_dir, fname)
                if os.path.exists(mask_path):
                    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

            if mask is None:
                continue

            # 调整大小
            img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, (img_size, img_size), interpolation=cv2.INTER_NEAREST)

            # 归一化
            img = img.astype(np.float32) / 255.0
            mask = (mask > 127).astype(np.float32)

            # 保存
            np.save(os.path.join(out_img_dir, f"{base_name}.npy"), img)
            np.save(os.path.join(out_mask_dir, f"{base_name}.npy"), mask)
            count += 1

        except Exception as e:
            print(f"  处理 {fname} 时出错: {e}")
            continue

    print(f"[Kvasir] ✓ 完成! 保存 {count} 张到 {output_dir}")
    return count


def prepare_cvc(raw_dir, output_dir, img_size=352):
    """准备CVC-ClinicDB数据集"""

    # 自动检测目录
    img_dir, mask_dir = find_image_mask_dirs(raw_dir, 'cvc')

    if img_dir is None or mask_dir is None:
        print(f"[CVC] ❌ 无法找到图像或mask目录")
        print(f"  尝试的路径: {raw_dir}")
        print(f"  检测到的图像目录: {img_dir}")
        print(f"  检测到的mask目录: {mask_dir}")
        return 0

    print(f"[CVC] 图像目录: {img_dir}")
    print(f"[CVC] Mask目录: {mask_dir}")

    out_img_dir = os.path.join(output_dir, "images")
    out_mask_dir = os.path.join(output_dir, "masks")
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_mask_dir, exist_ok=True)

    files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.tif', '.png', '.jpg', '.bmp', '.jpeg'))]
    print(f"[CVC] 找到 {len(files)} 张图像")

    if len(files) == 0:
        print(f"  ❌ 没有找到图像文件！请检查目录: {img_dir}")
        return 0

    count = 0
    for fname in tqdm(files, desc="[CVC] 处理中"):
        try:
            # 读取图像 (BGR -> RGB)
            img_path = os.path.join(img_dir, fname)
            img = cv2.imread(img_path)
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # 尝试多种mask文件名
            base_name = os.path.splitext(fname)[0]
            mask = None

            for ext in ['.tif', '.png', '.jpg', '.bmp']:
                mask_path = os.path.join(mask_dir, base_name + ext)
                if os.path.exists(mask_path):
                    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                    break

            if mask is None:
                continue

            # 调整大小
            img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, (img_size, img_size), interpolation=cv2.INTER_NEAREST)

            # 归一化
            img = img.astype(np.float32) / 255.0
            mask = (mask > 127).astype(np.float32)

            # 保存 (添加cvc_前缀避免命名冲突)
            np.save(os.path.join(out_img_dir, f"cvc_{base_name}.npy"), img)
            np.save(os.path.join(out_mask_dir, f"cvc_{base_name}.npy"), mask)
            count += 1

        except Exception as e:
            print(f"  处理 {fname} 时出错: {e}")
            continue

    print(f"[CVC] ✓ 完成! 保存 {count} 张到 {output_dir}")
    return count


def main():
    """主函数"""

    # 自动找到项目根目录
    project_root = find_project_root()
    os.chdir(project_root)

    print("=" * 60)
    print("准备跨数据集实验数据 (RGB彩色)")
    print("=" * 60)
    print(f"项目根目录: {project_root}")

    RAW_DIR = os.path.join(project_root, "data_raw")
    OUTPUT_DIR = os.path.join(project_root, "data_cross")

    # 检查数据结构
    exists, kvasir_path, cvc_path = check_data_structure(RAW_DIR)

    if not exists:
        print("\n" + "=" * 60)
        print("❌ 错误: 找不到 data_raw 目录!")
        print("=" * 60)
        print("\n请确保数据目录结构如下:")
        print(f"""
{project_root}/
├── data_raw/
│   ├── Kvasir-SEG/
│   │   ├── images/
│   │   └── masks/
│   └── CVC-ClinicDB/
│       ├── Original/
│       └── Ground Truth/
├── models/
└── cross_dataset/
        """)
        return

    if kvasir_path is None or cvc_path is None:
        print("\n" + "=" * 60)
        print("❌ 错误: 无法找到数据集目录!")
        print("=" * 60)
        if kvasir_path is None:
            print("  - 缺少 Kvasir-SEG 目录")
        if cvc_path is None:
            print("  - 缺少 CVC-ClinicDB 目录")
        print("\n请检查 data_raw 目录下的文件夹名称")
        return

    print("\n" + "-" * 60)

    # 准备数据
    n_kvasir = prepare_kvasir(
        kvasir_path,
        os.path.join(OUTPUT_DIR, "kvasir_full")
    )

    print()

    n_cvc = prepare_cvc(
        cvc_path,
        os.path.join(OUTPUT_DIR, "cvc_full")
    )

    # 总结
    print("\n" + "=" * 60)
    if n_kvasir > 0 and n_cvc > 0:
        print("✅ 数据准备完成!")
        print(f"   Kvasir-SEG: {n_kvasir} 张")
        print(f"   CVC-ClinicDB: {n_cvc} 张")
        print(f"\n   输出目录: {OUTPUT_DIR}")
    else:
        print("⚠️ 数据准备不完整!")
        if n_kvasir == 0:
            print("   - Kvasir-SEG 处理失败")
        if n_cvc == 0:
            print("   - CVC-ClinicDB 处理失败")
    print("=" * 60)

    # 验证数据
    if n_kvasir > 0:
        verify_data(os.path.join(OUTPUT_DIR, "kvasir_full"), "Kvasir")
    if n_cvc > 0:
        verify_data(os.path.join(OUTPUT_DIR, "cvc_full"), "CVC")


def verify_data(data_dir, name):
    """验证数据"""
    img_dir = os.path.join(data_dir, "images")
    mask_dir = os.path.join(data_dir, "masks")

    if not os.path.exists(img_dir):
        return

    files = os.listdir(img_dir)[:2]
    print(f"\n[验证 {name}]")
    for f in files:
        img = np.load(os.path.join(img_dir, f))
        mask = np.load(os.path.join(mask_dir, f))
        print(f"  {f}: img={img.shape}, range=[{img.min():.2f}, {img.max():.2f}], "
              f"mask={mask.shape}, unique={np.unique(mask)}")


if __name__ == "__main__":
    main()