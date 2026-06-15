"""
动态叠加增强管线
根据配置构建训练和验证的 albumentations.Compose。
包含自定义局部反色增强操作。
"""

import random
from functools import partial
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2


# ---- 自定义增强：随机矩形区域反色 ----
def random_invert_patches(image, num_patches_min=1, num_patches_max=3,
                          max_patch_size=8, **kwargs):
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
        patch = image[y:y+ph, x:x+pw]
        image[y:y+ph, x:x+pw] = 255 - patch
    return image


# ---- 增强构建函数 ----
def build_train_transform(cfg: dict) -> A.Compose:
    """
    根据配置构建训练增强管线。
    从 cfg['augmentation'] 读取所有参数，概率独立触发。
    """
    aug_cfg = cfg['augmentation']

    transforms = []

    # 1. 几何变换
    if 'affine' in aug_cfg:
        affine = aug_cfg['affine']
        transforms.append(
            A.Affine(
                rotate=affine['rotate'],
                shear=affine['shear'],
                scale=affine['scale'],
                translate_percent=affine.get('translate_percent', [0.1, 0.1]),
                p=affine.get('probability', 0.5)
            )
        )

    if 'elastic' in aug_cfg:
        elastic = aug_cfg['elastic']
        transforms.append(
            A.ElasticTransform(
                alpha=elastic.get('alpha', 120),
                sigma=elastic.get('sigma', 15),
                p=elastic.get('probability', 0.4)
            )
        )

    # 2. 遮挡与线条
    if 'coarse_dropout' in aug_cfg:
        cd = aug_cfg['coarse_dropout']
        transforms.append(
            A.CoarseDropout(
                num_holes_range=cd['num_holes_range'],
                hole_height_range=cd['hole_height_range'],
                hole_width_range=cd['hole_width_range'],
                p=cd.get('probability', 0.5)
            )
        )

    if 'grid_dropout' in aug_cfg:
        gd = aug_cfg['grid_dropout']
        transforms.append(
            A.GridDropout(
                ratio=gd.get('ratio', 0.2),
                unit_size_range=gd['unit_size_range'],
                random_offset=gd.get('random_offset', True),
                p=gd.get('probability', 0.4)
            )
        )

    # 3. 像素退化
    if 'gaussian_blur' in aug_cfg:
        gb = aug_cfg['gaussian_blur']
        transforms.append(
            A.GaussianBlur(
                blur_limit=gb.get('blur_limit', [3, 7]),
                sigma_limit=gb.get('sigma_limit', [0.5, 3.0]),
                p=gb.get('probability', 0.3)
            )
        )

    if 'gauss_noise' in aug_cfg:
        gn = aug_cfg['gauss_noise']
        transforms.append(
            A.GaussNoise(
                std_range=gn['std_range'],
                p=gn.get('probability', 0.4)
            )
        )

    # 仅支持 3 通道，无法使用
    # if 'iso_noise' in aug_cfg:
    #     iso = aug_cfg['iso_noise']
    #     transforms.append(
    #         A.ISONoise(
    #             p=iso.get('probability', 0.2)
    #         )
    #     )

    # 4. 局部反色（自定义）
    if 'random_invert' in aug_cfg:
        ri = aug_cfg['random_invert']
        num_min = ri.get('num_patches_min', 1)
        num_max = ri.get('num_patches_max', 3)
        max_size = ri.get('max_patch_size', 8)
        transforms.append(
            A.Lambda(
                image=partial(random_invert_patches, num_patches_min=num_min, num_patches_max=num_max, max_patch_size=max_size),
                p=ri.get('probability', 0.3)
            )
        )

    # 5. 归一化与张量转换（必须放在最后）
    data_cfg = cfg['data']
    transforms.append(
        A.Normalize(
            mean=data_cfg['mean'],
            std=data_cfg['std']
        )
    )
    transforms.append(ToTensorV2())

    return A.Compose(transforms)


def build_val_transform(cfg: dict) -> A.Compose:
    """验证/测试增强管线：仅归一化和转张量，不做任何破坏性增强"""
    data_cfg = cfg['data']
    return A.Compose([
        A.Normalize(mean=data_cfg['mean'], std=data_cfg['std']),
        ToTensorV2()
    ])