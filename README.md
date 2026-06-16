# Robust MNIST — 抗极端叠加破坏的 MNIST 分类

本项目提供一套完整的训练与推理流程，用于在遭受严重叠加破坏（线条、遮挡、反色、形变、噪声等）的 MNIST 数字上实现高鲁棒性分类。

模型采用针对 28×28 灰度图定制的 **ConvNeXt V2 Femto**，训练策略为全监督动态叠加增强 + EMA，无需自监督预训练。  
该项目特别适用于**已知破坏类型但未知其组合**的泛化任务，通过穷举式数据增强和针对性弱点修补，模型在极难验证集上已超越人类水平（>82%准确率）。若允许半监督学习，利用目标域无标签数据微调可进一步提升至接近满分。


## 项目结构

```
robust_mnist/
├── config/
│   ├── default.yaml       # 训练/推理配置模板
│   └── preprocessing.yaml # 数据源配置模板
├── data/
│   └── processed/                  # 预处理后的 .npy 文件
├── src/
│   ├── models/                     # 模型定义
│   │   ├── convnextv2_femto.py     # 定制 ConvNeXt V2 Femto
│   │   └── blocks.py               # 基础模块
│   ├── data_utils/
│   │   ├── dataset.py              # 内存映射数据集类
│   │   ├── augmentations.py        # 训练/验证增强管线
│   │   └── preprocessing.py        # 各格式数据加载工具
│   ├── training/                   # 训练核心模块
│   │   ├── trainer.py              # 训练循环、EMA、验证
│   │   ├── loss.py                 # 交叉熵（含标签平滑）
│   │   └── optimizer.py            # 优化器与余弦退火调度
│   ├── inference/                  # 推理与后处理
│   │   ├── predictor.py            # TTA 推理与提交文件生成
│   │   └── tta.py                  # 测试时增强
│   └── utils/                      # 日志、检查点、随机种子
├── scripts/                        # 入口脚本
│   ├── preprocess.py               # 数据预处理
│   ├── train.py                    # 训练
│   ├── infer.py                    # 推理
│   ├── error_analysis.py           # 错误样本分析
│   └── visualize_aug.py            # 增强效果可视化
├── checkpoints/                    # 模型权重
├── logs/                           # TensorBoard 日志
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
matplotlib     # 增强可视化
```

## 准备配置

1. 复制模板文件：
   ```bash
   cp config/default_template.yaml config/default.yaml
   cp config/preprocessing_template.yaml config/preprocessing.yaml
   ```
2. 编辑 `config/preprocessing.yaml`，填入你的数据集本地路径（支持 IDX、npy 目录、ImageFolder、affNIST .mat 格式）。
3. 编辑 `config/default.yaml`，按需调整超参数和设备（`training.device`）。该文件包含完整的增强策略，详见下文。

## 数据预处理

将各种来源的 MNIST 风格数据统一转换为 28×28 单通道的 `.npy` 文件，便于训练时快速加载。

```bash
python scripts/preprocess.py --config config/preprocessing.yaml
```

输出：`data/processed/train_images.npy` 和 `data/processed/train_labels.npy`。  
（验证集需自行准备，可参考预处理脚本从特定数据源生成 `val_*.npy`）

## 训练

```bash
python scripts/train.py --config config/default.yaml
```

- 每个 epoch 从全量数据中随机采样 80,000 张图像（可在配置中调整 `train_samples_per_epoch`）。
- 训练过程每 5 个 epoch 在验证集上评估，自动保存最佳 EMA 模型至 `checkpoints/best_ema.pth`。
- 支持混合精度训练（`training.mixed_precision` 设为 `true` 并指定 `amp_dtype`），默认 FP32。
- 训练日志实时写入 `logs/`，可用 TensorBoard 监控：`tensorboard --logdir logs`。

### 增强管线说明

训练时动态叠加以下增强，概率独立触发（具体参数见 `default.yaml`）。**训练中保留了 15% 的干净样本**（`clean_prob: 0.15`），确保模型不遗忘原始数字特征。

- **几何形变**：大幅随机仿射变换（旋转、剪切、缩放、平移）和极强弹性变形，模拟扭曲、拉伸。
- **遮挡**：大块随机灰度遮挡 + 微小孔洞遮挡（应对“巧妙遮挡”）。
- **线条**：等距细线（`prison_bars`），模拟“牢笼”状规则条纹。
- **像素退化**：轻微高斯模糊 + 主力高斯噪声 + 极少量极端高斯噪声（强度达人类辨认极限）。
- **反色**：随机大面积矩形反色。
- **混合**：MixUp / CutMix 批量混合。

你可以使用增强可视化脚本检查增强效果：
```bash
python scripts/visualize_aug.py --config config/default.yaml --output analysis/aug_sample.png
```

## 推理

用训练好的模型对测试集进行预测（支持 TTA 自动生成多视图平均）：

```bash
python scripts/infer.py --config config/default.yaml \
    --checkpoint checkpoints/best_ema.pth --output submission.csv
```

默认使用 12 视图测试时增强（缩放、平移、旋转），禁止水平翻转（防 6/9 混淆）。可调整 `inference.tta` 参数。

## 错误分析与模型诊断

理解模型的短板，可对带标签的验证集运行错误分析：

```bash
python scripts/error_analysis.py --config config/default.yaml \
    --checkpoint checkpoints/best_ema.pth --output_dir analysis
```

脚本将：
- 对验证集推理，找出所有预测错误的样本。
- 将错误图像保存至 `analysis/error_images/`。（相关代码默认被注释）
- 生成 `error_report.csv` 记录真实标签、预测标签、原始索引。（相关代码默认被注释）
- 生成 `error_grid.html` 网页，以网格形式展示所有错误样本，方便快速发现系统性失败模式（如反色、扭曲等）。

## 配置参考

`default_template.yaml` 中所有参数均有注释。关键部分：

- **模型**：`model.stages` 可调整各阶段深度、核大小、下采样位置。
- **数据**：`data.train_samples_per_epoch` 控制每 epoch 样本量（减小可加快 epoch 节奏，便于调参）。
- **增强**：`augmentation.*.probability` 及各强度参数，可按需开启/关闭或微调。`clean_prob` 控制完全不增强的样本比例。
- **优化**：`optimizer.lr`、`scheduler`（余弦退火 + warmup）。
- **正则化**：`label_smoothing`、`ema_decay`。
- **推理**：TTA 视图数、扰动范围。

## 利用无标签目标域进行半监督适应（可选）

若能在提交前获取测试集的无标签图像，可采用伪标签微调大幅提升性能：

1. 用最佳模型对测试图像进行 TTA 推理，筛选高置信度样本（如 softmax > 0.95）作为伪标签。
2. 将伪标签数据与源域训练数据混合，对每个测试集分别微调模型（源域 80%，伪标签 20%，学习率 1e-5，20~30 epoch）。
3. 推理时使用对应微调模型，可进一步提升 1~5% 准确率。

此策略尤其适合目标域难度显著低于源域的情况（如从极难验证集 82% 跃升至较易测试集 99%）。

## 许可证

本项目采用 [MIT License](LICENSE)。

## 注意事项

- 本代码开源时已移除所有绝对路径、比赛名称、测试集文件名等敏感信息。请使用模板配置填入本地数据。
- 请勿将 `checkpoints/`、`logs/`、`data/processed/`、`data/raw/` 上传至版本控制，已在 `.gitignore` 中忽略。
- Intel GPU 用户训练前建议设置环境变量 `export IGC_EnableDPEmulation=1`，以确保某些算子的兼容性。