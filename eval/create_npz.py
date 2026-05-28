import os
import sys
import re
from glob import glob
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm


def natural_key(path_str: str):
    """
    用于自然排序的 key：按数字块排序，其余部分按字符串排序。
    例如：img_2.png < img_10.png
    """
    return [
        int(text) if text.isdigit() else text.lower()
        for text in re.split(r"(\d+)", Path(path_str).name)
    ]


def create_npz_from_images(image_dir, output_npz, resize=None):
    """
    将文件夹中的图片转换为 NPZ 文件，并按照文件名排序。
    resize: 可选的尺寸调整参数 (width, height)
    """
    print(f"扫描目录 {image_dir} 中的图片...")
    image_files = []
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        image_files.extend(glob(os.path.join(image_dir, "**", ext), recursive=True))

    if not image_files:
        print(f"错误：在目录 {image_dir} 中未找到图片文件")
        return False

    image_files = sorted(image_files, key=natural_key)
    print(f"找到 {len(image_files)} 张图片，按文件名排序后开始处理...")

    images = []
    for img_path in tqdm(image_files):
        try:
            img = Image.open(img_path).convert("RGB")
            if resize:
                img = img.resize(resize)
            images.append(np.array(img))
        except Exception as e:
            print(f"处理图片 {img_path} 时出错: {e}")

    if not images:
        print("错误：无法处理任何图片")
        return False

    images_array = np.stack(images)
    print(f"图片处理完成，数组形状: {images_array.shape}")

    np.savez_compressed(output_npz, arr_0=images_array)
    print(f"已保存 NPZ 文件到 {output_npz}")
    return True


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("用法: python create_npz.py <图片目录> <输出NPZ文件>")
        sys.exit(1)
    
    image_dir = sys.argv[1]
    output_npz = sys.argv[2]

    if not os.path.exists(image_dir):
        print(f"错误：目录 {image_dir} 不存在")
        sys.exit(1)

    success = create_npz_from_images(image_dir, output_npz)
    # 若想统一尺寸，可改为：
    # success = create_npz_from_images(image_dir, output_npz, resize=(256, 256))

    if not success:
        sys.exit(1)