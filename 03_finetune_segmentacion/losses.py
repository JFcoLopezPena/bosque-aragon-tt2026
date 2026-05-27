"""
Focal + Dice Loss con ROI para segmentación binaria de copas de árboles.
El loss se calcula SOLO dentro de la ROI = máscara GT dilatada N píxeles.
Imágenes sin copa (máscara todo cero) devuelven loss=0.0 sin gradiente.
"""
from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Herramienta de dilatación ────────────────────────────────────────────────

def dilate_masks(masks: torch.Tensor, dilation_px: int) -> torch.Tensor:
    """
    Dilata máscaras binarias [B, H, W] int64 por dilation_px píxeles.

    Returns:
        Tensor [B, H, W] float32 con la ROI dilatada (1.0 dentro, 0.0 fuera).
    """
    if dilation_px <= 0:
        return masks.float()

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2 * dilation_px + 1, 2 * dilation_px + 1),
    )
    B = masks.shape[0]
    out = torch.zeros_like(masks, dtype=torch.float32)
    for b in range(B):
        m = masks[b].cpu().numpy().astype(np.uint8)
        out[b] = torch.from_numpy(cv2.dilate(m, kernel).astype(np.float32))

    return out.to(masks.device)  # [B, H, W] float32


# ── Componentes de loss ──────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """Binary Focal Loss con ROI mask."""

    def __init__(self, gamma: float = 2.0, alpha: float = 0.75) -> None:
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha  # > 0.5 → más peso a la clase copa (positiva)

    def forward(
        self,
        logits:  torch.Tensor,   # [B, 1, H, W]
        targets: torch.Tensor,   # [B, H, W] int64 {0, 1}
        roi:     torch.Tensor,   # [B, H, W] float32 {0, 1}
    ) -> torch.Tensor:
        probs    = torch.sigmoid(logits[:, 0])          # [B, H, W]
        targets_f = targets.float()

        ce = F.binary_cross_entropy_with_logits(logits[:, 0], targets_f, reduction="none")
        pt = torch.where(targets_f == 1, probs, 1 - probs)

        alpha_t = torch.where(
            targets_f == 1,
            torch.full_like(targets_f, self.alpha),
            torch.full_like(targets_f, 1 - self.alpha),
        )
        loss = alpha_t * (1 - pt) ** self.gamma * ce  # [B, H, W]

        denom = roi.sum() + 1e-6
        return (loss * roi).sum() / denom


class DiceLoss(nn.Module):
    """Soft Dice Loss con ROI mask."""

    def __init__(self, smooth: float = 1.0) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(
        self,
        logits:  torch.Tensor,
        targets: torch.Tensor,
        roi:     torch.Tensor,
    ) -> torch.Tensor:
        probs     = torch.sigmoid(logits[:, 0])  # [B, H, W]
        targets_f = targets.float()

        p_roi = probs     * roi
        t_roi = targets_f * roi

        inter = (p_roi * t_roi).sum(dim=(1, 2))
        union = p_roi.sum(dim=(1, 2)) + t_roi.sum(dim=(1, 2))

        dice = (2 * inter + self.smooth) / (union + self.smooth)
        return (1 - dice).mean()


# ── Loss combinado ────────────────────────────────────────────────────────────

class CrownSegmentationLoss(nn.Module):
    """
    Loss = focal_weight * FocalLoss + dice_weight * DiceLoss.
    Aplicado únicamente dentro de la ROI (GT dilatada dilation_px px).
    Si no hay copa en el batch, devuelve 0.0 sin gradiente.
    """

    def __init__(
        self,
        focal_weight: float = 0.5,
        dice_weight:  float = 0.5,
        focal_gamma:  float = 2.0,
        focal_alpha:  float = 0.75,
        dilation_px:  int   = 5,
    ) -> None:
        super().__init__()
        assert abs(focal_weight + dice_weight - 1.0) < 1e-6, \
            "focal_weight + dice_weight debe sumar 1.0"

        self.focal_weight = focal_weight
        self.dice_weight  = dice_weight
        self.dilation_px  = dilation_px

        self.focal = FocalLoss(gamma=focal_gamma, alpha=focal_alpha)
        self.dice  = DiceLoss()

    def forward(
        self,
        logits:  torch.Tensor,   # [B, 1, H, W]
        targets: torch.Tensor,   # [B, H, W] int64 {0, 1}
    ) -> tuple[torch.Tensor, dict[str, float]]:
        roi = dilate_masks(targets, self.dilation_px)  # [B, H, W] float32

        if roi.sum() == 0:
            zero = logits.sum() * 0.0  # mantiene el grafo pero devuelve 0
            return zero, {"focal_loss": 0.0, "dice_loss": 0.0, "total_loss": 0.0}

        focal = self.focal(logits, targets, roi)
        dice  = self.dice(logits, targets, roi)
        total = self.focal_weight * focal + self.dice_weight * dice

        return total, {
            "focal_loss": focal.item(),
            "dice_loss":  dice.item(),
            "total_loss": total.item(),
        }


def build_loss(config: dict) -> CrownSegmentationLoss:
    lcfg = config.get("loss", {})
    return CrownSegmentationLoss(
        focal_weight=lcfg.get("focal_weight", 0.5),
        dice_weight=lcfg.get("dice_weight",  0.5),
        focal_gamma=lcfg.get("focal_gamma",  2.0),
        focal_alpha=lcfg.get("focal_alpha",  0.75),
        dilation_px=lcfg.get("roi_dilation_px", 5),
    )
