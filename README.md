## README.md

# Robust MNIST — 抗恶意叠加增强的 MNIST 分类

本项目旨在将 **不可见、具有极端叠加增强** 的 MNIST 测试集分类准确率推至 **99%+**。  
通过动态叠加破坏增强、定制 ConvNeXt V2 Femto 架构和指数移动平均等正则化手段，在单 GPU 上稳定训练出高泛化模型。

## 注意
默认配置中使用 Intel XPU 而非常见的 CUDA 加速。

## 技术路线

- **模型**: ConvNeXt V2 Femto (约 5M 参数)，适配 28×28 单通道输入
  - 仅 2 次下采样 (28→14→7)
  - 大核卷积 (7×7) 提供全图感受野，天然抵抗局部破坏
- **训练策略**: 全监督学习，无需自监督预训练
- **数据增强**: 在训练时动态叠加 **7 种恶意破坏**（仿射、弹性变形、遮挡、横竖线、高斯模糊、噪声、局部反色），同时混合 MixUp/CutMix，使训练分布直接覆盖未知测试分布
- **正则化**: EMA (0.9999)、Label Smoothing (0.1)、DropPath
- **推理**: 测试时增强 (TTA) — 12 个几何扰动视图平均

## 项目结构

```
robust_mnist/
├── config/
│   ├── default.yaml          # 训练/推理完整配置
│   └── preprocessing.yaml    # 数据预处理数据源配置
├── data/
│   └── processed/            # 预处理生成的 .npy 文件
├── src/
│   ├── models/               # 模型定义
│   │   ├── convnextv2_femto.py
│   │   └── blocks.py
│   ├── data_utils/           # 数据处理框架
│   │   ├── dataset.py        # 内存映射数据集类
│   │   ├── augmentations.py  # 训练/验证增强管线
│   │   └── preprocessing.py  # 各格式数据加载工具
│   ├── training/             # 训练核心模块
│   │   ├── trainer.py
│   │   ├── loss.py
│   │   └── optimizer.py
│   ├── inference/            # 推理模块
│   │   ├── predictor.py
│   │   └── tta.py
│   └── utils/                # 工具 (日志/检查点/种子)
├── scripts/                  # 执行入口
│   ├── preprocess.py         # 数据预处理
│   ├── train.py              # 训练
│   └── infer.py              # 推理
├── requirements.txt
└── README.md
```

## 环境要求

- Python 3.8+
- PyTorch 2.0+ (CUDA 推荐)
- 其他依赖: `pip install -r requirements.txt`

可选依赖
```
tensorboard
```

## 使用方法

### 1. 数据预处理

将所有原始数据集（MNIST、MNIST-C、affNIST、EMNIST、Gen-Hard 等）统一转换为 28×28 单通道 `.npy` 文件。

支持的格式：ubyte mat [数字]/*.(png jpg 等) [增强名称]/train_images(labels).npy

参考 `config_example/` 下的 `train_preprocessing.yaml` 和 `val_preprocessing.yaml` 。

**配置数据源** — 编辑 `config/*_preprocessing.yaml`，填写各数据集的本地路径与类型。  
**运行**:
```bash
python scripts/preprocess.py --config config/train_preprocessing.yaml
python scripts/preprocess.py --config config/val_preprocessing.yaml
```
输出: `data/processed/train_images.npy`  `data/processed/train_labels.npy` `data/processed/val_images.npy`  `data/processed/val_labels.npy`

### 2. 训练

**配置训练** — 可根据需要修改 `config/default.yaml`（已包含所有超参）。  
**启动训练**:
```bash
python scripts/train.py --config config/default.yaml
```
训练过程默认会每 10 个 epoch 在验证集上评估一次，并自动保存最佳 EMA 模型至 `checkpoints/best_ema.pth`。

### 3. 推理

**生成测试集预测**:
```bash
python scripts/infer.py --config config/default.yaml --checkpoint checkpoints/best_ema.pth --output submission.csv
```
脚本自动执行 12 视图 TTA，输出 CSV 文件。

## 配置说明

所有关键超参集中在 `config/default.yaml`，包括:
- `model`：模型架构参数
- `data`：数据路径、采样量、batch size
- `augmentation`：每种增强的概率与强度
- `training`：epoch 数、验证频率、设备
- `optimizer` / `scheduler`：学习率、warmup、余弦退火
- `regularization`：label smoothing、EMA 衰减
- `inference`：TTA 参数和输出路径

## 性能预期

在预留的验证集（未参与训练的叠加破坏样本）上，该方案通常可达到 **99.0% - 99.4%** 准确率。  
TTA 与模型集成 (训练 2-3 个不同种子) 可将结果进一步推至 99.5%+。

## 引用与致谢

- ConvNeXt V2 (https://arxiv.org/abs/2301.00808)
- albumentations (https://albumentations.ai/)
- MNIST-C (https://github.com/google-research/mnist-c)