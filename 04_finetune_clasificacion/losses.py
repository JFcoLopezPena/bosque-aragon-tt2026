"""
Loss combinado para clasificacion fitosanitaria + segmentacion de copa.

Loss total = 0.6 * cls_loss + 0.4 * seg_loss

cls_loss = 0.5 * CrossEntropy(class_weights) + 0.5 * FocalLoss(class_weights)
seg_loss = 0.5 * BinaryFocal(ROI) + 0.5 * Dice(ROI)
"""
from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Helpers ───────────────────────────────────────────────────────────────────

def dilate_masks(masks: torch.Tensor, dilation_px: int) -> torch.Tensor:
    """Dilata mascaras binarias [B, H, W] con kernel eliptico."""
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * dilation_px + 1, 2 * dilation_px + 1)
    )
    out = []
    for b in range(masks.shape[0]):
        m = masks[b].cpu().numpy().astype(np.uint8)
        d = cv2.dilate(m, kernel)
        out.append(torch.from_numpy(d.astype(np.float32)))
    return torch.stack(out).to(masks.device)


# ── Segmentation losses (same as R1) ─────────────────────────────────────────

class BinaryFocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha: float = 0.5) -> None:
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(
        self,
        logits:  torch.Tensor,          # [B, 1, H, W]
        targets: torch.Tensor,          # [B, 1, H, W] float {0,1}
        roi:     torch.Tensor | None = None,  # [B, 1, H, W] float {0,1}
    ) -> torch.Tensor:
        if logits.dim()  == 4: logits  = logits[:, 0]
        if targets.dim() == 4: targets = targets[:, 0]
        if roi is not None and roi.dim() == 4: roi = roi[:, 0]

        bce     = F.binary_cross_entropy_with_logits(logits, targets.float(), reduction="none")
        prob    = torch.sigmoid(logits)
        pt      = targets.float() * prob + (1 - targets.float()) * (1 - prob)
        alpha_t = targets.float() * self.alpha + (1 - targets.float()) * (1 - self.alpha)
        focal   = alpha_t * (1 - pt) ** self.gamma * bce

        if roi is not None:
            n = roi.sum()
            if n == 0:
                return logits.sum() * 0.0
            return (focal * roi).sum() / n

        return focal.mean()


class DiceLoss(nn.Module):
    def forward(
        self,
        logits:  torch.Tensor,
        targets: torch.Tensor,
        roi:     torch.Tensor | None = None,
    ) -> torch.Tensor:
        if logits.dim()  == 4: logits  = logits[:, 0]
        if targets.dim() == 4: targets = targets[:, 0]
        if roi is not None and roi.dim() == 4: roi = roi[:, 0]

        prob    = torch.sigmoid(logits)
        targets = targets.float()

        if roi is not None:
            prob    = prob    * roi
            targets = targets * roi

        inter = (prob * targets).sum()
        union = prob.sum() + targets.sum()
        return 1.0 - (2.0 * inter + 1e-8) / (union + 1e-8)


# ── Classification losses ─────────────────────────────────────────────────────

class FocalLossMultiClass(nn.Module):
    """Focal Loss para clasificacion N-clases con class weights opcionales."""
    def __init__(self, gamma: float = 2.0, weight: torch.Tensor | None = None) -> None:
        super().__init__()
        self.gamma = gamma
        if weight is not None:
            self.register_buffer("weight", weight)
        else:
            self.weight = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits: [B, C], targets: [B] int64
        ce   = F.cross_entropy(logits, targets, weight=self.weight, reduction="none")
        prob = F.softmax(logits, dim=1)
        pt   = prob.gather(1, targets.unsqueeze(1)).squeeze(1)
        return ((1 - pt) ** self.gamma * ce).mean()


# ── Combined loss ─────────────────────────────────────────────────────────────

class CombinedLoss(nn.Module):
    def __init__(self, config: dict, class_weights: torch.Tensor | None = None) -> None:
        super().__init__()
        lc = config["loss"]

        self.focal_weight = lc["focal_weight"]
        self.dice_weight  = lc["dice_weight"]
        self.use_roi      = lc.get("use_roi_mask", True)
        self.dilation_px  = lc.get("roi_dilation_px", 5)
        self.cls_weight   = lc.get("cls_weight", 0.6)
        self.seg_weight   = lc.get("seg_weight", 0.4)

        self.seg_focal = BinaryFocalLoss(gamma=lc["focal_gamma"], alpha=lc["focal_alpha"])
        self.seg_dice  = DiceLoss()
        self.cls_ce    = nn.CrossEntropyLoss(weight=class_weights)
        self.cls_focal = FocalLossMultiClass(gamma=lc["focal_gamma"], weight=class_weights)

    def forward(
        self,
        logits_cls:  torch.Tensor,   # [B, 2]
        logits_mask: torch.Tensor,   # [B, 1, H, W]
        labels:      torch.Tensor,   # [B] int64
        masks:       torch.Tensor,   # [B, 1, H, W] float {0,1}
    ) -> tuple[torch.Tensor, dict]:

        # ── Segmentation loss (ROI-based, same as R1) ──────────────────────
        mask_bin = (masks[:, 0] > 0.5).long()
        roi = None
        if self.use_roi:
            roi_hw = dilate_masks(mask_bin, self.dilation_px)
            if roi_hw.sum() > 0:
                roi = roi_hw.unsqueeze(1)  # [B, 1, H, W]

        focal_seg = self.seg_focal(logits_mask, masks, roi)
        dice_seg  = self.seg_dice( logits_mask, masks, roi)
        seg_loss  = self.focal_weight * focal_seg + self.dice_weight * dice_seg

        # ── Classification loss ────────────────────────────────────────────
        ce_cls   = self.cls_ce(logits_cls, labels)
        foc_cls  = self.cls_focal(logits_cls, labels)
        cls_loss = 0.5 * ce_cls + 0.5 * foc_cls

        # ── Combined ───────────────────────────────────────────────────────
        total = self.cls_weight * cls_loss + self.seg_weight * seg_loss

        return total, {
            "total_loss": total.item(),
            "cls_loss":   cls_loss.item(),
            "seg_loss":   seg_loss.item(),
            "focal_seg":  focal_seg.item(),
            "dice_seg":   dice_seg.item(),
        }
