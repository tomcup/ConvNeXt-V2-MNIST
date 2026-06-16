"""
快速可视化增强管线
用法：
    python scripts/visualize_aug.py --config config/default.yaml --output analysis/aug_sample.png
"""

import argparse
import sys
from pathlib import Path

import yaml
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from src.data_utils.dataset import PreloadedDataset
from src.data_utils.augmentations import build_train_transform

# 确保可导入src
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main():
    parser = argparse.ArgumentParser(description="可视化训练增强效果")
    parser.add_argument("--config", default="config/default.yaml", help="配置文件路径")
    parser.add_argument("--output", default="aug_sample.png", help="输出图片路径")
    parser.add_argument("--num_samples", type=int, default=16, help="展示图像数量")
    args = parser.parse_args()

    # 加载配置
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # 构建训练增强（纯增强，不包含归一化到tensor的步骤？我们已包含Normalize和ToTensorV2）
    transform = build_train_transform(cfg)

    # 加载数据集（使用训练数据，任意取一个子集）
    data_cfg = cfg["data"]
    dataset = PreloadedDataset(
        images_path=data_cfg["train_images_path"],
        labels_path=data_cfg["train_labels_path"],
        transform=transform,
        soft_labels=False,
    )
    # 取少量样本，不打乱，快速加载
    loader = DataLoader(
        dataset, batch_size=args.num_samples, shuffle=True, num_workers=0
    )

    # 获取一个batch
    batch = next(iter(loader))
    images = batch[0]  # (B, 1, 28, 28) 归一化后的tensor

    # 反归一化以便可视化（mean=0.5, std=0.5 -> 转回[0,1]）
    images = images * 0.5 + 0.5
    # 限制在[0,1]
    images = images.clamp(0, 1)

    # 拼接成网格
    grid = torch.cat([img for img in images.cpu()], dim=2)  # 水平拼接
    # 如果要多行多列，可以用torchvision.utils.make_grid
    import torchvision.utils as vutils

    grid = vutils.make_grid(images.cpu(), nrow=4, padding=2, normalize=False)
    # 转为matplotlib可显示格式 (C, H, W) -> (H, W, C)
    grid_np = grid.permute(1, 2, 0).numpy()
    plt.figure(figsize=(10, 10))
    plt.imshow(grid_np, cmap="gray")
    plt.axis("off")
    plt.title("Augmented Training Samples")
    plt.savefig(args.output, bbox_inches="tight", dpi=150)
    print(f"增强样本图已保存至 {args.output}")


if __name__ == "__main__":
    main()
