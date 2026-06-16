"""
推理预测模块
加载最佳 EMA 模型，对测试集执行 TTA 推理，输出提交文件。
"""

import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset

from .tta import TTAPredictor


class TestDataset(Dataset):
    """测试数据集，从 .npy 内存映射文件中读取（无标签）"""

    def __init__(self, npy_path: str, mean: float, std: float):
        self.data = np.load(npy_path, mmap_mode="r")  # shape (N, 28, 28) uint8
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx]  # (28,28) uint8 [0,255]
        img = img.astype(np.float32) / 255.0  # [0,1]
        img = (img - self.mean) / self.std  # 归一化到约[-1,1]
        img = torch.from_numpy(img).unsqueeze(0)  # (1,28,28)
        return img


def run_inference(cfg: dict):
    """
    执行完整的推理流程：
    1. 加载模型
    2. 加载测试数据
    3. 使用 TTA 预测
    4. 保存 CSV 结果
    """
    # ---------- 设备 ----------
    device = torch.device(cfg.get("training", {}).get("device", "cuda"))

    # ---------- 加载模型 ----------
    checkpoint_path = cfg["inference"]["checkpoint"]
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # 动态导入模型架构（避免循环依赖）
    from ..models.convnextv2_femto import ConvNeXtV2FemtoMNIST

    model = ConvNeXtV2FemtoMNIST()
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # ---------- 数据加载 ----------
    data_cfg = cfg["data"]
    test_path = data_cfg.get("test_images_path", None)
    if test_path is None or not os.path.isfile(test_path):
        raise FileNotFoundError(
            "Test dataset not found. Please run preprocessing for test data."
        )
    mean = data_cfg["mean"][0]
    std = data_cfg["std"][0]
    test_dataset = TestDataset(test_path, mean, std)
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg["inference"]["batch_size"],
        shuffle=False,
        num_workers=data_cfg.get("num_workers", 4),
        pin_memory=data_cfg.get("pin_memory", True),
    )

    # ---------- TTA 配置 ----------
    tta_cfg = cfg["inference"]["tta"]
    tta = TTAPredictor(
        num_views=tta_cfg.get("num_views", 12),
        scale_range=tta_cfg.get("scale_range", (0.9, 1.1)),
        translate=tta_cfg.get("translate", 2),
        rotation=tta_cfg.get("rotation", 10.0),
        horizontal_flip=tta_cfg.get("horizontal_flip", False),
    )

    # ---------- 预测 ----------
    predictions = []
    with torch.no_grad():
        for images in test_loader:
            images = images.to(device)
            probs = tta.predict(model, images)  # (B, num_classes)
            preds = probs.argmax(dim=-1)  # (B,)
            predictions.extend(preds.cpu().tolist())

    # ---------- 保存结果 ----------
    output_file = cfg["inference"]["output_file"]
    # 生成 ID（用 0-based 索引）
    ids = list(range(len(predictions)))
    df = pd.DataFrame({"ID": ids, "Label": predictions})
    df.to_csv(output_file, index=False)
    print(f"Inference completed. Predictions saved to {output_file}")
