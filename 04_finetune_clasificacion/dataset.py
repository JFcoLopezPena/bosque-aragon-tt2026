"""
Dataset para clasificacion fitosanitaria binaria (SANO / ENFERMO) - Ronda 2.
Lee pares (imagen RGB, mascara R1, label) desde splits_r2.json.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import albumentations as A
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

CLASS_TO_IDX = {"sano": 0, "enfermo": 1}
IDX_TO_CLASS = {0: "sano", 1: "enfermo"}


def find_image(directory: Path, stem: str) -> Optional[Path]:
    for ext in (".png", ".jpg", ".jpeg"):
        p = directory / (stem + ext)
        if p.exists():
            return p
    return None


def build_transforms(config: dict, split: str, image_size: int) -> A.Compose:
    aug = config.get("augmentation", {})

    spatial = [A.Resize(image_size, image_size)]

    if split == "train":
        if aug.get("horizontal_flip", 0) > 0:
            spatial.append(A.HorizontalFlip(p=aug["horizontal_flip"]))
        if aug.get("vertical_flip", 0) > 0:
            spatial.append(A.VerticalFlip(p=aug["vertical_flip"]))
        if aug.get("random_rotate_90", 0) > 0:
            spatial.append(A.RandomRotate90(p=aug["random_rotate_90"]))
        if aug.get("elastic_transform", 0) > 0:
            spatial.append(A.ElasticTransform(p=aug["elastic_transform"]))
        if aug.get("grid_distortion", 0) > 0:
            spatial.append(A.GridDistortion(p=aug["grid_distortion"]))

    pixel = []
    if split == "train":
        cj = aug.get("color_jitter", {})
        if cj:
            pixel.append(A.ColorJitter(
                brightness=cj.get("brightness", 0),
                contrast=cj.get("contrast", 0),
                saturation=cj.get("saturation", 0),
                hue=0,
                p=0.5,
            ))

    return A.Compose(spatial + pixel, additional_targets={"mask": "mask"})


class TreeClassificationDataset(Dataset):
    def __init__(
        self,
        data_root: Path,
        split_data: dict,    # {"sano": [tree_id, ...], "enfermo": [...]}
        transform: A.Compose,
        image_size: int = 512,
    ) -> None:
        self.transform  = transform
        self.image_size = image_size
        self.items: list[tuple[Path, Path, int]] = []  # (rgb_path, mask_path, label)

        for cls_name, tree_ids in split_data.items():
            label    = CLASS_TO_IDX[cls_name]
            img_dir  = data_root / "images" / cls_name
            mask_dir = data_root / "masks"  / cls_name

            for tree_id in tree_ids:
                rgb_path  = find_image(img_dir, tree_id)
                mask_path = mask_dir / (tree_id + ".png")
                if rgb_path is None or not mask_path.exists():
                    continue
                self.items.append((rgb_path, mask_path, label))

        n_sano    = sum(1 for _, _, l in self.items if l == 0)
        n_enfermo = sum(1 for _, _, l in self.items if l == 1)
        total     = len(self.items)
        pct_s     = 100 * n_sano    / max(total, 1)
        pct_e     = 100 * n_enfermo / max(total, 1)
        print(f"  [{total} muestras]  sano={n_sano} ({pct_s:.1f}%)  enfermo={n_enfermo} ({pct_e:.1f}%)")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        rgb_path, mask_path, label = self.items[idx]

        img = cv2.imread(str(rgb_path))
        if img is None:
            img = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            mask = np.zeros((self.image_size, self.image_size), dtype=np.uint8)

        out  = self.transform(image=img, mask=mask)
        img  = out["image"]
        mask = out["mask"]

        img_t  = torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1)  # [3,H,W]
        mask_t = torch.from_numpy((mask > 127).astype(np.float32)).unsqueeze(0)      # [1,H,W]

        return img_t, mask_t, label


def create_dataloaders(config: dict) -> tuple[DataLoader, DataLoader, DataLoader]:
    data_root   = Path(config["data"]["data_root"])
    splits_file = Path(config["data"]["splits_file"])
    image_size  = config["training"]["image_size"]
    batch_size  = config["training"]["batch_size"]
    num_workers = config["data"]["num_workers"]
    pin_memory  = config["data"]["pin_memory"]

    with open(splits_file, "r", encoding="utf-8") as f:
        splits = json.load(f)

    loaders: dict[str, DataLoader] = {}
    for split in ("train", "val", "test"):
        tf = build_transforms(config, split, image_size)
        print(f"[Dataset] {split}:", end="")
        ds = TreeClassificationDataset(data_root, splits[split], tf, image_size)
        loaders[split] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    return loaders["train"], loaders["val"], loaders["test"]
