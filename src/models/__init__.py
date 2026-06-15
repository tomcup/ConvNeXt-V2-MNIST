from .convnextv2_femto import ConvNeXtV2FemtoMNIST

__all__ = ["ConvNeXtV2FemtoMNIST", "build_model"]


def build_model(config: dict):
    """
    根据配置字典构建模型
    当前支持: convnextv2_femto_mnist
    """
    model_name = config["model"]["name"]
    if model_name == "convnextv2_femto_mnist":
        return ConvNeXtV2FemtoMNIST(
            in_channels=config["model"]["in_channels"],
            num_classes=config["model"]["num_classes"],
            stem_out_channels=config["model"]["stem"]["out_channels"],
            head_dropout=config["model"]["head"]["dropout"],
        )
    else:
        raise ValueError(f"Unknown model name: {model_name}")