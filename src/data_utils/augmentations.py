"""
动态叠加增强管线
根据配置构建训练和验证的 albumentations.Compose。
包含自定义局部反色增强操作。
"""

import random
from functools import partial
import albumentations as A
from albumentations.pytorch import ToTensorV2


# ---- 自定义增强：随机矩形区域反色 ----
def random_invert_patches(
    image, num_patches_min=1, num_patches_max=3, max_patch_size=8, **kwargs
):
    """
    在图像上随机选取若干矩形区域进行像素反转 (255 - x)
    Args:
        image: numpy 数组，shape (H, W)，值域 [0, 255]
        num_patches_min, num_patches_max: 矩形数量范围
        max_patch_size: 矩形最大边长
    Returns:
        处理后的图像 (numpy array)
    """
    h, w = image.shape[:2]
    num_patches = random.randint(num_patches_min, num_patches_max)
    for _ in range(num_patches):
        ph = random.randint(2, min(max_patch_size, h))
        pw = random.randint(2, min(max_patch_size, w))
        y = random.randint(0, h - ph)
        x = random.randint(0, w - pw)
        patch = image[y : y + ph, x : x + pw]
        image[y : y + ph, x : x + pw] = 255 - patch
    return image


# 结构性反色：基于网格划分随机反色
def grid_based_invert(image, min_div=2, max_div=4, invert_prob=0.5, **kwargs):
    """
    通过横竖分割线将图像划分为网格，
    然后随机选取部分区域进行反色。
    """
    import random

    h, w = image.shape[:2]
    # 生成分割边界（不画线，只用于确定区域坐标）
    num_h = random.randint(min_div, max_div)
    num_v = random.randint(min_div, max_div)
    # 边界位置（像素坐标），避免太靠边
    h_borders = sorted([random.randint(4, h - 4) for _ in range(num_h)])
    v_borders = sorted([random.randint(4, w - 4) for _ in range(num_v)])

    # 完整边界列表：0, 分割线, 图像末端
    y_borders = [0] + h_borders + [h]
    x_borders = [0] + v_borders + [w]

    # 对每个矩形区域随机反色
    for i in range(len(y_borders) - 1):
        for j in range(len(x_borders) - 1):
            if random.random() < invert_prob:
                y1, y2 = y_borders[i], y_borders[i + 1]
                x1, x2 = x_borders[j], x_borders[j + 1]
                image[y1:y2, x1:x2] = 255 - image[y1:y2, x1:x2]
    return image


# 牢笼干扰：在图像上添加随机等距细线，模拟扫描条纹或规则纹理。
def add_prison_bars(
    image,
    orientation="random",
    bar_width_range=(1, 2),
    spacing=(5, 9),
    jitter_max=2,
    color_range=(180, 255),
    opacity_range=(0.7, 1.0),
    offset_random=True,
    **kwargs,
):
    """
    添加随机等距细线（“牢笼”干扰），模拟扫描条纹或规则纹理。
    支持随机起始偏移、间距抖动、线宽变化、灰度变化和不透明度变化。
    """
    h, w = image.shape[:2]

    # 决定方向
    if orientation == "random":
        orient = random.choice(["horizontal", "vertical"])
    else:
        orient = orientation

    # 基础间距
    base_space = random.randint(spacing[0], spacing[1])

    # 随机起始偏移
    if offset_random:
        offset = random.randint(0, base_space - 1) if base_space > 1 else 0
    else:
        offset = 0

    # 遍历绘制线条
    pos = offset
    while pos < (h if orient == "horizontal" else w):
        # 加入抖动，但确保位置不至于跳回太远（不越界）
        jitter = random.randint(-jitter_max, jitter_max) if jitter_max > 0 else 0
        actual_pos = pos + jitter
        # 限制在图像范围内
        if orient == "horizontal":
            actual_pos = max(0, min(actual_pos, h - 1))
        else:
            actual_pos = max(0, min(actual_pos, w - 1))

        # 线宽
        bw = random.randint(bar_width_range[0], bar_width_range[1])
        # 颜色 (灰度)
        color = random.randint(color_range[0], color_range[1])
        # 不透明度
        opacity = random.uniform(opacity_range[0], opacity_range[1])

        # 绘制线条（混合）
        if orient == "horizontal":
            y1 = actual_pos
            y2 = min(y1 + bw, h)  # 不超出边界
            # 原像素值线性混合
            image[y1:y2, :] = (
                image[y1:y2, :] * (1.0 - opacity) + color * opacity
            ).astype(image.dtype)
        else:
            x1 = actual_pos
            x2 = min(x1 + bw, w)
            image[:, x1:x2] = (
                image[:, x1:x2] * (1.0 - opacity) + color * opacity
            ).astype(image.dtype)

        # 下一个位置（不含抖动的基础位置递增）
        pos += base_space

    return image


# ---- 增强构建函数 ----
def build_train_transform(cfg: dict) -> A.Compose:
    """
    根据配置构建训练增强管线。
    从 cfg['augmentation'] 读取所有参数，概率独立触发。
    """
    aug_cfg = cfg["augmentation"]
    data_cfg = cfg["data"]
    clean_prob = aug_cfg.get("clean_prob", 0.15)  # 默认 15%

    # 干净概率：以一定几率仅归一化，不应用任何破坏增强
    if random.random() < clean_prob:
        return A.Compose(
            [A.Normalize(mean=data_cfg["mean"], std=data_cfg["std"]), ToTensorV2()]
        )

    transforms = []

    # 1. 几何变换
    if "affine" in aug_cfg:
        affine = aug_cfg["affine"]
        transforms.append(
            A.Affine(
                rotate=affine["rotate"],
                shear=affine["shear"],
                scale=affine["scale"],
                translate_percent=affine.get("translate_percent", [0.1, 0.1]),
                p=affine.get("probability", 0.5),
            )
        )

    if "elastic" in aug_cfg:
        elastic = aug_cfg["elastic"]
        transforms.append(
            A.ElasticTransform(
                alpha=elastic.get("alpha", 120),
                sigma=elastic.get("sigma", 15),
                p=elastic.get("probability", 0.4),
            )
        )

    # 2. 遮挡与线条
    if "coarse_dropout" in aug_cfg:
        cd = aug_cfg["coarse_dropout"]
        transforms.append(
            A.CoarseDropout(
                num_holes_range=cd["num_holes_range"],
                hole_height_range=cd["hole_height_range"],
                hole_width_range=cd["hole_width_range"],
                fill=cd.get("fill", 0),
                p=cd.get("probability", 0.5),
            )
        )

    # 微小孔洞遮挡
    if "tiny_dropout" in aug_cfg:
        td = aug_cfg["tiny_dropout"]
        transforms.append(
            A.CoarseDropout(
                num_holes_range=td["num_holes_range"],
                hole_height_range=td["hole_height_range"],
                hole_width_range=td["hole_width_range"],
                fill=td.get("fill", 0),
                p=td.get("probability", 0.5),
            )
        )

    # if 'grid_dropout' in aug_cfg:
    #     gd = aug_cfg['grid_dropout']
    #     transforms.append(
    #         A.GridDropout(
    #             ratio=gd.get('ratio', 0.2),
    #             unit_size_range=gd['unit_size_range'],
    #             random_offset=gd.get('random_offset', True),
    #             fill=gd.get('fill', 0),
    #             p=gd.get('probability', 0.4)
    #         )
    #     )

    if "prison_bars" in aug_cfg:
        pb = aug_cfg["prison_bars"]
        transforms.append(
            A.Lambda(
                image=partial(
                    add_prison_bars,
                    orientation=pb.get("orientation", "random"),
                    bar_width_range=pb.get("bar_width_range", [1, 2]),
                    spacing=pb.get("spacing", [5, 9]),
                    jitter_max=pb.get("jitter_max", 2),
                    color_range=pb.get("color_range", [180, 255]),
                    opacity_range=pb.get("opacity_range", [0.7, 1.0]),
                    offset_random=pb.get("offset_random", True),
                ),
                p=pb.get("probability", 0.3),
            )
        )

    # 3. 像素退化
    if "gaussian_blur" in aug_cfg:
        gb = aug_cfg["gaussian_blur"]
        transforms.append(
            A.GaussianBlur(
                blur_limit=gb.get("blur_limit", [3, 7]),
                sigma_limit=gb.get("sigma_limit", [0.5, 3.0]),
                p=gb.get("probability", 0.3),
            )
        )

    if "gauss_noise" in aug_cfg:
        gn = aug_cfg["gauss_noise"]
        transforms.append(
            A.GaussNoise(std_range=gn["std_range"], p=gn.get("probability", 0.4))
        )

    if "extreme_gauss_noise" in aug_cfg:
        eg = aug_cfg["extreme_gauss_noise"]
        transforms.append(
            A.GaussNoise(std_range=eg["std_range"], p=eg.get("probability", 0.08))
        )

    # 4. 局部反色（自定义）
    if "random_invert" in aug_cfg:
        ri = aug_cfg["random_invert"]
        num_min = ri.get("num_patches_min", 1)
        num_max = ri.get("num_patches_max", 3)
        max_size = ri.get("max_patch_size", 8)
        transforms.append(
            A.Lambda(
                image=partial(
                    random_invert_patches,
                    num_patches_min=num_min,
                    num_patches_max=num_max,
                    max_patch_size=max_size,
                ),
                p=ri.get("probability", 0.3),
            )
        )

    # if 'grid_invert' in aug_cfg:
    #     gi = aug_cfg['grid_invert']
    #     transforms.append(
    #         A.Lambda(
    #             image=partial(grid_based_invert,
    #                           min_div=gi.get('min_div', 2),
    #                           max_div=gi.get('max_div', 4),
    #                           invert_prob=gi.get('invert_prob', 0.5)),
    #             p=gi.get('probability', 0.25)
    #         )
    #     )

    # 5. 归一化与张量转换（必须放在最后）
    data_cfg = cfg["data"]
    transforms.append(A.Normalize(mean=data_cfg["mean"], std=data_cfg["std"]))
    transforms.append(ToTensorV2())

    return A.Compose(transforms)


def build_val_transform(cfg: dict) -> A.Compose:
    """验证/测试增强管线：仅归一化和转张量，不做任何破坏性增强"""
    data_cfg = cfg["data"]
    return A.Compose(
        [A.Normalize(mean=data_cfg["mean"], std=data_cfg["std"]), ToTensorV2()]
    )
