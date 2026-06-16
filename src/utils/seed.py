"""
随机种子固定模块
确保实验可复现
"""

import random
import os
import numpy as np
import torch


def set_seed(seed: int = 42):
    """固定 Python、NumPy、PyTorch 的随机种子"""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.xpu.manual_seed(
        seed
    )  # It's safe to call this function if XPU is not available; in that case, it is silently ignored.

    # 确保卷积等操作在相同输入下产生相同输出（可能影响性能）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
