"""
Dataset para segmentación binaria de copas de árboles con SAM.
Entrada: RGB .png  +  máscara binaria .png (255=copa, 0=fondo)
Salida:  imagen [3, H, W] float32 en [0,1]  +  máscara [H, W] int64 {0,1}
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import List, Tuple, Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A


class TreeCrownDataset(Dataset):
    """
    Dataset de copas de árboles para segmentación binaria.

    Args:
        image_paths: lista de rutas a imágenes RGB .png
        mask_paths:  lista de rutas a máscaras binarias (mismo orden)
        image_size:  tamaño de salida cuadrado (dataset; SAM recibe 1024 en train)
        augment:     True solo en split de entrenamiento
        aug_config:  sub-dict del config YAML con probabilidades
    """

    def __init__(
        self,
        image_paths: List[Path],
        mask_paths:  List[Path],
        image_size:  int = 512,
        augment:     bool = False,
        aug_config:  Optional[dict] = None,
    ) -> None:
        assert len(image_paths) == len(mask_paths)
        self.image_paths = image_paths
        self.mask_paths  = mask_paths
        self.image_size  = image_size
        self.augment     = augment
        self.aug_config  = aug_config or {}
        self.transform   = self._build_transform()

    def _build_transform(self) -> A.Compose:
        cfg = self.aug_config
        t = []

        if self.augment:
            t += [
                A.HorizontalFlip(p=cfg.get("horizontal_flip", 0.5)),
                A.VerticalFlip(p=cfg.get("vertical_flip", 0.5)),
                A.RandomRotate90(p=cfg.get("random_rotate90", 0.7)),
                A.ElasticTransform(p=cfg.get("elastic_transform", 0.3)),
                A.RandomBrightnessContrast(
                    brightness_limit=0.25,
                    contrast_limit=0.25,
                    p=cfg.get("brightness_contrast", 0.4),
                ),
                A.Blur(blur_limit=3, p=cfg.get("blur", 0.2)),
                A.RandomResizedCrop(
                    size=(self.image_size, self.image_size),
                    scale=(
                        cfg.get("crop_scale_min", 0.7),
                        cfg.get("crop_scale_max", 1.0),
                    ),
                    ratio=(0.85, 1.15),
                    p=cfg.get("random_resized_crop", 0.3),
                ),
            ]

        t.append(A.Resize(self.image_size, self.image_size, interpolation=cv2.INTER_LINEAR))

        return A.Compose(t, additional_targets={"mask": "mask"})

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            image: [3, H, W] float32 en [0, 1] (sin normalización de canal)
            mask:  [H, W] int64 {0=fondo, 1=copa}
        """
        # Leer imagen como RGB uint8
        bgr = cv2.imread(str(self.image_paths[idx]))
        if bgr is None:
            raise IOError(f"No se pudo leer imagen: {self.image_paths[idx]}")
        image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)  # [H, W, 3] uint8

        # Leer máscara y binarizar
        mask_raw = cv2.imread(str(self.mask_paths[idx]), cv2.IMREAD_GRAYSCALE)
        if mask_raw is None:
            raise IOError(f"No se pudo leer máscara: {self.mask_paths[idx]}")
        mask = (mask_raw > 127).astype(np.uint8)  # {0, 1}

        # Igualar tamaños si difieren
        if mask.shape[:2] != image.shape[:2]:
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]),
                              interpolation=cv2.INTER_NEAREST)

        aug = self.transform(image=image, mask=mask)
        image_aug = aug["image"]  # [H, W, 3] uint8
        mask_aug  = aug["mask"]   # [H, W] uint8

        # [0, 255] → [0, 1]  (SAM normaliza internamente con sus propios stats)
        image_t = torch.from_numpy(image_aug.astype(np.float32) / 255.0).permute(2, 0, 1)
        mask_t  = torch.from_numpy(mask_aug.copy()).long()

        return image_t, mask_t


def split_dataset(
    images_dir:  Path,
    masks_dir:   Path,
    train_ratio: float = 0.70,
    val_ratio:   float = 0.15,
    seed:        int   = 42,
) -> Tuple[List[Path], List[Path], List[Path], List[Path], List[Path], List[Path]]:
    """Divide pares imagen/máscara en train / val / test con seed fija."""
    image_paths = sorted(p for p in images_dir.iterdir() if p.suffix.lower() == ".png")
    if not image_paths:
        raise ValueError(f"No se encontraron .png en {images_dir}")

    paired, missing = [], []
    for img in image_paths:
        mask = masks_dir / img.name
        if mask.exists():
            paired.append((img, mask))
        else:
            missing.append(img.name)

    if missing:
        print(f"[Dataset] WARN: {len(missing)} imágenes sin máscara, descartadas")

    random.seed(seed)
    random.shuffle(paired)

    n       = len(paired)
    n_train = int(n * train_ratio)
    n_val   = int(n * val_ratio)

    def unzip(lst):
        if not lst:
            return [], []
        a, b = zip(*lst)
        return list(a), list(b)

    ti, tm = unzip(paired[:n_train])
    vi, vm = unzip(paired[n_train:n_train + n_val])
    xi, xm = unzip(paired[n_train + n_val:])

    print(f"[Dataset] {n} parejas  ->  train={len(ti)} | val={len(vi)} | test={len(xi)}")
    return ti, tm, vi, vm, xi, xm


def create_dataloaders(
    config: dict,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Crea DataLoaders train/val/test desde config."""
    images_dir = Path(config["data"]["images_dir"])
    masks_dir  = Path(config["data"]["masks_dir"])
    aug_cfg    = config.get("augmentation", {})
    img_size   = config["training"]["image_size"]
    batch      = config["training"]["batch_size"]
    workers    = config["data"]["num_workers"]
    pin_mem    = config["data"]["pin_memory"]
    seed       = config["training"]["seed"]

    ti, tm, vi, vm, xi, xm = split_dataset(
        images_dir, masks_dir,
        train_ratio=config["data"]["train_ratio"],
        val_ratio=config["data"]["val_ratio"],
        seed=seed,
    )

    train_ds = TreeCrownDataset(ti, tm, image_size=img_size, augment=True,  aug_config=aug_cfg)
    val_ds   = TreeCrownDataset(vi, vm, image_size=img_size, augment=False, aug_config=aug_cfg)
    test_ds  = TreeCrownDataset(xi, xm, image_size=img_size, augment=False, aug_config=aug_cfg)

    train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True,
                              num_workers=workers, pin_memory=pin_mem, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch, shuffle=False,
                              num_workers=workers, pin_memory=pin_mem)
    test_loader  = DataLoader(test_ds,  batch_size=batch, shuffle=False,
                              num_workers=workers, pin_memory=pin_mem)

    return train_loader, val_loader, test_loader
