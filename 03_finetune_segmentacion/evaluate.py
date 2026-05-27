"""
Evaluación del modelo fine-tuneado sobre el test set.
Métricas: IoU, F1, Precision, Recall, Boundary IoU
Salida: JSON con métricas + visualizaciones (imagen | GT | predicción)

Uso:
    python evaluate.py
    python evaluate.py --config path/to/config.yaml
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import autocast
import yaml

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))

from dataset      import create_dataloaders, TreeCrownDataset
from model        import SegmentationSAM


# ── Métricas ──────────────────────────────────────────────────────────────────

def compute_metrics(
    preds:   torch.Tensor,   # [N, H, W] bool/int
    targets: torch.Tensor,   # [N, H, W] int64
) -> Dict[str, float]:
    preds   = preds.bool()
    targets = targets.bool()

    tp = (preds  &  targets).sum().float()
    fp = (preds  & ~targets).sum().float()
    fn = (~preds &  targets).sum().float()
    tn = (~preds & ~targets).sum().float()

    iou       = tp / (tp + fp + fn + 1e-8)
    precision = tp / (tp + fp + 1e-8)
    recall    = tp / (tp + fn + 1e-8)
    f1        = 2 * precision * recall / (precision + recall + 1e-8)
    accuracy  = (tp + tn) / (tp + fp + fn + tn + 1e-8)

    return {
        "iou":       iou.item(),
        "f1":        f1.item(),
        "precision": precision.item(),
        "recall":    recall.item(),
        "accuracy":  accuracy.item(),
    }


def boundary_iou(
    preds:   torch.Tensor,   # [N, H, W] int64
    targets: torch.Tensor,   # [N, H, W] int64
    dilation_px: int = 3,
) -> float:
    """
    Boundary IoU: IoU calculado solo sobre los píxeles de frontera.
    La frontera se obtiene dilatando la máscara y restando el interior (erosión).
    """
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * dilation_px + 1, 2 * dilation_px + 1)
    )

    def get_boundary(mask_np: np.ndarray) -> np.ndarray:
        dilated  = cv2.dilate(mask_np, kernel)
        eroded   = cv2.erode(mask_np, kernel)
        boundary = dilated - eroded
        return (boundary > 0).astype(np.uint8)

    N  = preds.shape[0]
    tp = fp = fn = 0
    for n in range(N):
        p_np = preds[n].cpu().numpy().astype(np.uint8)
        t_np = targets[n].cpu().numpy().astype(np.uint8)
        bp   = get_boundary(p_np)
        bt   = get_boundary(t_np)
        # Boundary de ambos juntos
        union_b = np.maximum(bp, bt)
        tp += (bp & bt).sum()
        fp += (bp & ~bt.astype(bool)).sum()
        fn += (~bp.astype(bool) & bt).sum()

    return float(tp) / (tp + fp + fn + 1e-8)


# ── Visualizaciones ───────────────────────────────────────────────────────────

def save_visualizations(
    images:  torch.Tensor,   # [B, 3, H, W] ∈ [0, 1]
    targets: torch.Tensor,   # [B, H, W]
    preds:   torch.Tensor,   # [B, H, W] bool
    out_dir: Path,
    prefix:  str = "",
    n_show:  int = 6,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    B = min(images.shape[0], n_show)

    for b in range(B):
        img = (images[b].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        gt  = targets[b].cpu().numpy().astype(np.uint8) * 255
        pr  = preds[b].cpu().numpy().astype(np.uint8)   * 255

        # Overlay predicción sobre imagen (verde=TP, rojo=FP, azul=FN)
        img_rgb = img.copy()
        tp_mask = (preds[b] & targets[b].bool()).cpu().numpy()
        fp_mask = (preds[b] & ~targets[b].bool()).cpu().numpy()
        fn_mask = (~preds[b] & targets[b].bool()).cpu().numpy()

        overlay = img_rgb.copy()
        overlay[tp_mask] = (overlay[tp_mask] * 0.5 + np.array([0, 200, 0]) * 0.5).astype(np.uint8)
        overlay[fp_mask] = (overlay[fp_mask] * 0.5 + np.array([200, 0, 0]) * 0.5).astype(np.uint8)
        overlay[fn_mask] = (overlay[fn_mask] * 0.5 + np.array([0, 0, 200]) * 0.5).astype(np.uint8)

        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        axes[0].imshow(img_rgb);  axes[0].set_title("Imagen RGB"); axes[0].axis("off")
        axes[1].imshow(gt, cmap="gray"); axes[1].set_title("GT Máscara"); axes[1].axis("off")
        axes[2].imshow(pr, cmap="gray"); axes[2].set_title("Predicción"); axes[2].axis("off")
        axes[3].imshow(overlay);         axes[3].set_title("TP=verde FP=rojo FN=azul"); axes[3].axis("off")

        plt.tight_layout()
        fig.savefig(out_dir / f"{prefix}sample_{b:03d}.png", dpi=100, bbox_inches="tight")
        plt.close(fig)


# ── Pipeline de evaluación ────────────────────────────────────────────────────

def evaluate(config_path: str = "config.yaml") -> None:
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Eval] Dispositivo: {device}")

    ckpt_dir    = Path(config["paths"]["checkpoint_dir"])
    results_dir = Path(config["paths"]["results_dir"])
    viz_dir     = results_dir / "visualizations"
    results_dir.mkdir(parents=True, exist_ok=True)

    best_path = ckpt_dir / "best_model.pth"
    if not best_path.exists():
        raise FileNotFoundError(f"Checkpoint no encontrado: {best_path}")

    print(f"[Eval] Cargando modelo desde {best_path} ...")
    model = SegmentationSAM.load_checkpoint(
        ft_checkpoint=best_path,
        sam_checkpoint=config["model"]["checkpoint_path"],
    ).to(device)
    model.eval()

    print("[Eval] Cargando test set ...")
    _, _, test_loader = create_dataloaders(config)

    all_logits, all_masks  = [], []
    all_images_for_viz     = []
    use_amp = config["training"]["mixed_precision"]

    with torch.no_grad():
        for images, masks in test_loader:
            images = images.to(device)
            masks  = masks.to(device)

            images_1024 = F.interpolate(images, size=(1024, 1024),
                                        mode="bilinear", align_corners=False)
            H, W = masks.shape[-2], masks.shape[-1]

            with autocast("cuda", enabled=use_amp):
                logits = model(images_1024, output_size=(H, W))

            all_logits.append(logits.cpu())
            all_masks.append(masks.cpu())
            if len(all_images_for_viz) < config["logging"]["num_viz_samples"]:
                all_images_for_viz.append(images.cpu())

    all_logits = torch.cat(all_logits)   # [N, 1, H, W]
    all_masks  = torch.cat(all_masks)    # [N, H, W]
    all_preds  = (torch.sigmoid(all_logits[:, 0]) > 0.65).long()

    metrics = compute_metrics(all_preds, all_masks)
    metrics["boundary_iou"] = boundary_iou(all_preds, all_masks, dilation_px=3)

    print(f"\n{'='*50}")
    print("[Eval] Resultados en test set:")
    for k, v in metrics.items():
        print(f"  {k:15s}: {v:.4f}")
    print(f"{'='*50}\n")

    json_path = results_dir / "test_metrics.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"[Eval] Métricas guardadas: {json_path}")

    # Visualizaciones
    if config["logging"]["save_visualizations"]:
        all_images_viz = torch.cat(all_images_for_viz)[: config["logging"]["num_viz_samples"]]
        n              = all_images_viz.shape[0]
        save_visualizations(
            images=all_images_viz,
            targets=all_masks[:n],
            preds=all_preds[:n].bool(),
            out_dir=viz_dir,
            n_show=n,
        )
        print(f"[Eval] Visualizaciones guardadas: {viz_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    os.chdir(Path(__file__).parent)
    evaluate(args.config)
