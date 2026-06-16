import torch

import sys
from pathlib import Path

from src.models.convnextv2_femto import ConvNeXtV2FemtoMNIST

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# 加载你的最佳权重
model = ConvNeXtV2FemtoMNIST()
checkpoint = torch.load("checkpoints/best_ema.pth", map_location="cpu")
model.load_state_dict(checkpoint)
model.eval()

example_input = torch.randn(1, 1, 28, 28)

# 使用 trace 导出
with torch.no_grad():
    traced_model = torch.jit.trace(model, example_input)

# 保存为 TorchScript 文件
traced_model.save("convnextv2_femto_scripted.pt")
