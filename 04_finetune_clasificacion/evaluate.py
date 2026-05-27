"""
Evaluacion del modelo de clasificacion fitosanitaria sobre el test set.

Metricas: Accuracy, F1/Precision/Recall por clase, F1 macro/weighted,
          Matriz de confusion, IoU de segmentacion (verificacion vs R1=0.615)

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
from typing import Dict

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

from dataset import create_dataloaders, IDX_TO_CLASS
from model   import ClassificationSAM

R1_IOU_BASELINE = 0.615


# ── Metricas ──────────────────────────────────────────────────────────────────

def compute_all_metrics(
    preds:   torch.Tensor,   # [N] int64
    targets: torch.Tensor,   # [N] int64
) -> Dict[str, float]:
    acc    = (preds == targets).float().mean().item()
    total  = len(targets)
    counts = {c: (targets == c).sum().item() for c in (0, 1)}

    metrics: Dict[str, float] = {"accuracy": acc, "n_total": total}
    f1s = {}

    for cls_idx, cls_name in ((0, "sano"), (1, "enfermo")):
        tp = ((preds == cls_idx) & (targets == cls_idx)).sum().float()
        fp = ((preds == cls_idx) & (targets != cls_idx)).sum().float()
        fn = ((preds != cls_idx) & (targets == cls_idx)).sum().float()
        tn = ((preds != cls_idx) & (targets != cls_idx)).sum().float()

        prec = (tp / (tp + fp + 1e-8)).item()
        rec  = (tp / (tp + fn + 1e-8)).item()
        f1   = 2 * prec * rec / (prec + rec + 1e-8)
        spec = (tn / (tn + fp + 1e-8)).item()

        metrics[f"precision_{cls_name}"] = prec
        metrics[f"recall_{cls_name}"]    = rec
        metrics[f"f1_{cls_name}"]        = f1
        metrics[f"specificity_{cls_name}"] = spec
        metrics[f"n_{cls_name}"]         = counts[cls_idx]
        f1s[cls_idx] = f1

    metrics["f1_macro"]    = (f1s[0] + f1s[1]) / 2
    metrics["f1_weighted"] = (f1s[0] * counts[0] + f1s[1] * counts[1]) / max(total, 1)
    return metrics


def compute_iou(
    logits_mask: torch.Tensor,   # [N, 1, H, W]
    masks:       torch.Tensor,   # [N, 1, H, W]
    threshold:   float = 0.65,
) -> float:
    preds   = (torch.sigmoid(logits_mask[:, 0]) > threshold).long()
    targets = (masks[:, 0] > 0.5).long()
    tp = ((preds == 1) & (targets == 1)).sum().float()
    fp = ((preds == 1) & (targets == 0)).sum().float()
    fn = ((preds == 0) & (targets == 1)).sum().float()
    return (tp / (tp + fp + fn + 1e-8)).item()


def build_confusion_matrix(
    preds:   torch.Tensor,
    targets: torch.Tensor,
    n_cls:   int = 2,
) -> np.ndarray:
    cm = np.zeros((n_cls, n_cls), dtype=np.int64)
    for t, p in zip(targets.numpy(), preds.numpy()):
        cm[t][p] += 1
    return cm


def plot_confusion_matrix(cm: np.ndarray, class_names: list[str], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, fontsize=12)
    ax.set_yticklabels(class_names, fontsize=12)
    ax.set_xlabel("Prediccion", fontsize=12)
    ax.set_ylabel("Real", fontsize=12)
    ax.set_title("Matriz de Confusion (test set)", fontsize=13)

    thresh = cm.max() / 2.0
    for i, j in np.ndindex(cm.shape):
        ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                fontsize=14, fontweight="bold",
                color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ── Visualizaciones ───────────────────────────────────────────────────────────

def save_visualizations(
    test_loader: torch.utils.data.DataLoader,
    model:       ClassificationSAM,
    device:      torch.device,
    out_dir:     Path,
    n_show:      int = 8,
    use_amp:     bool = True,
    image_size:  int = 512,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    model.eval()

    # Collect up to n_show samples (one at a time from first batches)
    samples = []
    for images, masks, labels in test_loader:
        for b in range(images.shape[0]):
            samples.append((images[b], masks[b], labels[b].item()))
            if len(samples) >= n_show:
                break
        if len(samples) >= n_show:
            break

    colors = {0: np.array([0, 200, 0]), 1: np.array([200, 0, 0])}
    cls_names = {0: "SANO", 1: "ENFERMO"}

    for idx, (img_t, mask_t, true_label) in enumerate(samples):
        img_1024 = F.interpolate(
            img_t.unsqueeze(0), (1024, 1024), mode="bilinear", align_corners=False
        ).to(device)

        with torch.no_grad():
            with autocast("cuda", enabled=use_amp):
                logits_cls, _ = model(img_1024)
        pred_label = logits_cls.argmax(dim=1).item()
        probs      = torch.softmax(logits_cls, dim=1).cpu()[0]

        img_np  = (img_t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        mask_np = mask_t[0].numpy()

        # Overlay: color mask region by predicted class
        overlay   = img_np.copy()
        mask_bool = mask_np > 0.5
        color     = colors[pred_label]
        overlay[mask_bool] = (overlay[mask_bool] * 0.5 + color * 0.5).astype(np.uint8)

        mask_vis = (mask_np * 255).astype(np.uint8)
        correct  = pred_label == true_label

        fig, axes = plt.subplots(1, 4, figsize=(18, 4))
        axes[0].imshow(img_np)
        axes[0].set_title(f"RGB | Real: {cls_names[true_label]}", fontsize=10)
        axes[0].axis("off")

        axes[1].imshow(mask_vis, cmap="gray")
        axes[1].set_title("Mascara R1", fontsize=10)
        axes[1].axis("off")

        # Prediction panel
        bg_color = (0.13, 0.55, 0.13) if pred_label == 0 else (0.75, 0.15, 0.15)
        axes[2].set_facecolor(bg_color)
        axes[2].text(
            0.5, 0.5,
            f"{cls_names[pred_label]}\n{probs[pred_label]:.2f}",
            transform=axes[2].transAxes,
            ha="center", va="center", fontsize=16, fontweight="bold", color="white",
        )
        axes[2].set_title("Prediccion", fontsize=10)
        axes[2].set_xticks([]); axes[2].set_yticks([])

        axes[3].imshow(overlay)
        axes[3].set_title("Overlay (verde=SANO, rojo=ENFERMO)", fontsize=10)
        axes[3].axis("off")

        status = "CORRECTO" if correct else "ERROR"
        fig.suptitle(status, fontsize=13,
                     color="green" if correct else "red", fontweight="bold")
        plt.tight_layout()
        fig.savefig(out_dir / f"sample_{idx:03d}.png", dpi=100, bbox_inches="tight")
        plt.close(fig)

    print(f"[Eval] Visualizaciones guardadas: {out_dir}")


# ── Pipeline de evaluacion ────────────────────────────────────────────────────

def evaluate(config_path: str = "config.yaml") -> None:
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results_dir = Path(config["paths"]["results_dir"])
    ckpt_dir    = Path(config["paths"]["checkpoint_dir"])
    viz_dir     = results_dir / "visualizations"
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Eval] Dispositivo: {device}")

    best_path = ckpt_dir / "best_model.pth"
    if not best_path.exists():
        raise FileNotFoundError(f"Checkpoint no encontrado: {best_path}")

    print(f"[Eval] Cargando modelo desde {best_path} ...")
    model = ClassificationSAM.load_checkpoint(
        ft_checkpoint=best_path,
        sam_checkpoint=config["model"]["sam_checkpoint"],
        unfreeze_encoder_blocks=config["model"].get("unfreeze_encoder_blocks", 2),
        num_classes=config["model"]["num_classes"],
    ).to(device)
    model.eval()

    # Load metadata from checkpoint
    ckpt     = torch.load(str(best_path), map_location="cpu", weights_only=False)
    metadata = ckpt.get("metadata", {})
    if metadata:
        print(f"[Eval] Checkpoint: epoch={metadata.get('epoch','?')} "
              f"val_F1_enfermo={metadata.get('f1_enfermo', '?'):.4f}")

    print("[Eval] Cargando test set ...")
    _, _, test_loader = create_dataloaders(config)
    n_test = len(test_loader.dataset)
    print(f"[Eval] Test set: {n_test} muestras")

    use_amp    = config["training"]["mixed_precision"]
    image_size = config["training"]["image_size"]

    all_logits_cls  = []
    all_logits_mask = []
    all_labels      = []
    all_masks       = []

    with torch.no_grad():
        for images, masks, labels in test_loader:
            images = images.to(device)
            masks  = masks.to( device)
            labels = labels.to(device)

            images_1024 = F.interpolate(images, (1024, 1024), mode="bilinear", align_corners=False)

            with autocast("cuda", enabled=use_amp):
                logits_cls, logits_mask = model(images_1024, output_size=(image_size, image_size))

            all_logits_cls.append( logits_cls.cpu())
            all_logits_mask.append(logits_mask.cpu())
            all_labels.append(     labels.cpu())
            all_masks.append(      masks.cpu())

    all_logits_cls  = torch.cat(all_logits_cls)
    all_logits_mask = torch.cat(all_logits_mask)
    all_labels      = torch.cat(all_labels)
    all_masks       = torch.cat(all_masks)
    all_preds       = all_logits_cls.argmax(dim=1)

    # ── Classification metrics ────────────────────────────────────────────────
    metrics = compute_all_metrics(all_preds, all_labels)

    # ── Segmentation IoU ──────────────────────────────────────────────────────
    iou_seg = compute_iou(all_logits_mask, all_masks)
    metrics["iou_segmentation"] = iou_seg

    # ── Confusion matrix ──────────────────────────────────────────────────────
    cm = build_confusion_matrix(all_preds, all_labels)
    cm_path = results_dir / "confusion_matrix.png"
    plot_confusion_matrix(cm, ["SANO", "ENFERMO"], cm_path)
    print(f"[Eval] Confusion matrix guardada: {cm_path}")

    # ── Print results ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("[Eval] RESULTADOS EN TEST SET")
    print("="*60)
    print(f"  N total           : {metrics['n_total']}")
    print(f"  Accuracy          : {metrics['accuracy']:.4f}")
    print()
    print(f"  {'Clase':<12} {'F1':>6} {'Precision':>10} {'Recall':>8} {'N':>6}")
    print(f"  {'-'*44}")
    for cls_name in ("sano", "enfermo"):
        n_cls = metrics[f"n_{cls_name}"]
        f1    = metrics[f"f1_{cls_name}"]
        prec  = metrics[f"precision_{cls_name}"]
        rec   = metrics[f"recall_{cls_name}"]
        print(f"  {cls_name:<12} {f1:>6.4f} {prec:>10.4f} {rec:>8.4f} {n_cls:>6}")
    print()
    print(f"  F1 macro          : {metrics['f1_macro']:.4f}")
    print(f"  F1 weighted       : {metrics['f1_weighted']:.4f}")
    print()
    print(f"  IoU segmentacion  : {iou_seg:.4f}  (baseline R1: {R1_IOU_BASELINE})")
    print("="*60)

    # ── Objetivo / diagnostico ────────────────────────────────────────────────
    f1e = metrics["f1_enfermo"]
    if f1e >= 0.65:
        print(f"\n  OBJETIVO F1_enfermo >= 0.65: ALCANZADO ({f1e:.4f})")
    else:
        print(f"\n  OBJETIVO F1_enfermo >= 0.65: NO alcanzado (max={f1e:.4f})")
        print("  Diagnostico:")
        if metrics["recall_enfermo"] < 0.60:
            print("    - Recall bajo: modelo no detecta suficientes enfermos")
            print("      -> Incrementar class_weights[1] o usar mas augmentacion")
        if metrics["precision_enfermo"] < 0.60:
            print("    - Precision baja: demasiados falsos positivos enfermo")
            print("      -> Reducir class_weights[1] o ajustar threshold de decision")
        if iou_seg < R1_IOU_BASELINE - 0.02:
            print("    - IoU de segmentacion degradado: decoder perdio calidad R1")
            print("      -> Incrementar seg_weight en loss o reducir lr del decoder")

    if iou_seg < R1_IOU_BASELINE:
        print(f"\n  WARN: IoU segmentacion ({iou_seg:.4f}) < baseline R1 ({R1_IOU_BASELINE})")
        print("        Considera: seg_weight mayor, lr decoder menor, menos epochs")
    else:
        print(f"\n  IoU segmentacion OK ({iou_seg:.4f} >= {R1_IOU_BASELINE})")

    print("="*60 + "\n")

    # ── Confusion matrix valores ──────────────────────────────────────────────
    print(f"  Confusion matrix (filas=real, columnas=pred):")
    print(f"            SANO    ENFERMO")
    print(f"  SANO    {cm[0,0]:6d}   {cm[0,1]:6d}")
    print(f"  ENFERMO {cm[1,0]:6d}   {cm[1,1]:6d}\n")

    # ── Export JSON ───────────────────────────────────────────────────────────
    out_json = results_dir / "test_metrics.json"
    export   = {k: (float(v) if not isinstance(v, int) else v) for k, v in metrics.items()}
    export["confusion_matrix"] = cm.tolist()
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(export, f, indent=2)
    print(f"[Eval] Metricas guardadas: {out_json}")

    # ── Visualizaciones ───────────────────────────────────────────────────────
    if config["logging"]["save_visualizations"]:
        save_visualizations(
            test_loader=test_loader,
            model=model,
            device=device,
            out_dir=viz_dir,
            n_show=config["logging"]["num_viz_samples"],
            use_amp=use_amp,
            image_size=image_size,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    os.chdir(Path(__file__).parent)
    evaluate(args.config)
