"""
Loop de entrenamiento — Fine-tuning de SAM para segmentación de copas de árboles.

Uso:
    python train.py
    python train.py --config path/to/config.yaml
"""
from __future__ import annotations

import argparse
import csv
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
import yaml

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))

from dataset import create_dataloaders
from losses  import build_loss, CrownSegmentationLoss
from model   import SegmentationSAM


# ── Utilidades ────────────────────────────────────────────────────────────────

def load_config(path: str | Path = "config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def compute_seg_metrics(
    logits:  torch.Tensor,   # [B, 1, H, W]
    targets: torch.Tensor,   # [B, H, W] int64
) -> Dict[str, float]:
    """IoU, F1, Precision, Recall de copa (clase 1)."""
    preds = (torch.sigmoid(logits[:, 0]) > 0.65).long()

    tp = ((preds == 1) & (targets == 1)).sum().float()
    fp = ((preds == 1) & (targets == 0)).sum().float()
    fn = ((preds == 0) & (targets == 1)).sum().float()

    iou       = tp / (tp + fp + fn + 1e-8)
    precision = tp / (tp + fp + 1e-8)
    recall    = tp / (tp + fn + 1e-8)
    f1        = 2 * precision * recall / (precision + recall + 1e-8)

    return {
        "iou":       iou.item(),
        "f1":        f1.item(),
        "precision": precision.item(),
        "recall":    recall.item(),
    }


def save_curves(history: dict, path: Path) -> None:
    epochs = list(range(1, len(history["train_loss"]) + 1))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, history["train_loss"], label="train")
    axes[0].plot(epochs, history["val_loss"],   label="val")
    axes[0].set_title("Loss"); axes[0].legend(); axes[0].set_xlabel("Epoch")

    axes[1].plot(epochs, history["val_iou"], color="tab:green")
    axes[1].set_title("Val IoU (copa)"); axes[1].set_xlabel("Epoch")
    axes[1].set_ylim(0, 1)

    plt.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ── Train / Validate ──────────────────────────────────────────────────────────

def train_one_epoch(
    model:     SegmentationSAM,
    loader:    torch.utils.data.DataLoader,
    optimizer: AdamW,
    criterion: CrownSegmentationLoss,
    scaler:    GradScaler,
    device:    torch.device,
    config:    dict,
    epoch:     int,
) -> Dict[str, float]:
    model.train()
    total_loss = total_focal = total_dice = 0.0
    n          = len(loader)
    log_every  = config["logging"]["log_every_n_batches"]
    use_amp    = config["training"]["mixed_precision"]
    clip_norm  = config["training"]["gradient_clip"]
    t0         = time.time()

    for i, (images, masks) in enumerate(loader):
        images = images.to(device, non_blocking=True)  # [B, 3, 512, 512] ∈ [0,1]
        masks  = masks.to(device,  non_blocking=True)  # [B, 512, 512] int64

        # Upscale a 1024×1024 para el encoder de SAM
        images_1024 = F.interpolate(
            images, size=(1024, 1024), mode="bilinear", align_corners=False
        )
        H, W = masks.shape[-2], masks.shape[-1]

        optimizer.zero_grad(set_to_none=True)

        with autocast("cuda", enabled=use_amp):
            logits = model(images_1024, output_size=(H, W))   # [B, 1, H, W]
            loss, ld = criterion(logits, masks)

        if loss.requires_grad:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.get_trainable_params(), clip_norm)
            scaler.step(optimizer)
            scaler.update()

        total_loss  += ld["total_loss"]
        total_focal += ld["focal_loss"]
        total_dice  += ld["dice_loss"]

        if (i + 1) % log_every == 0 or i == n - 1:
            elapsed = time.time() - t0
            avg = total_loss / (i + 1)
            print(f"  Epoch {epoch:3d} | Batch {i+1:4d}/{n} | "
                  f"Loss={avg:.4f} Focal={ld['focal_loss']:.4f} "
                  f"Dice={ld['dice_loss']:.4f} | {elapsed:.0f}s")

    return {
        "train_loss":  total_loss  / n,
        "train_focal": total_focal / n,
        "train_dice":  total_dice  / n,
    }


@torch.no_grad()
def validate(
    model:     SegmentationSAM,
    loader:    torch.utils.data.DataLoader,
    criterion: CrownSegmentationLoss,
    device:    torch.device,
    config:    dict,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    all_logits, all_masks = [], []
    use_amp = config["training"]["mixed_precision"]

    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks  = masks.to(device,  non_blocking=True)

        images_1024 = F.interpolate(images, size=(1024, 1024),
                                    mode="bilinear", align_corners=False)
        H, W = masks.shape[-2], masks.shape[-1]

        with autocast("cuda", enabled=use_amp):
            logits = model(images_1024, output_size=(H, W))
            _, ld  = criterion(logits, masks)

        total_loss += ld["total_loss"]
        all_logits.append(logits.cpu())
        all_masks.append(masks.cpu())

    all_logits = torch.cat(all_logits)
    all_masks  = torch.cat(all_masks)

    metrics = compute_seg_metrics(all_logits, all_masks)
    metrics["val_iou"]  = metrics.pop("iou")
    metrics["val_loss"] = total_loss / len(loader)
    return metrics


# ── Pipeline principal ────────────────────────────────────────────────────────

def train(config_path: str = "config.yaml") -> None:
    config = load_config(config_path)
    set_seed(config["training"]["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Train] Dispositivo: {device}")
    if torch.cuda.is_available():
        print(f"[Train] GPU: {torch.cuda.get_device_name(0)}")

    ckpt_dir = Path(config["paths"]["checkpoint_dir"])
    logs_dir = Path(config["paths"]["logs_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    print("[Train] Creando DataLoaders ...")
    train_loader, val_loader, _ = create_dataloaders(config)

    print("[Train] Cargando modelo ...")
    n_unfreeze = config["training"].get("unfreeze_encoder_blocks", 0)
    model = SegmentationSAM(
        checkpoint_path=config["model"]["checkpoint_path"],
        unfreeze_encoder_blocks=n_unfreeze,
    ).to(device)

    lr      = config["training"]["learning_rate"]
    wd      = config["training"]["weight_decay"]

    # LR diferencial: encoder descongelado aprende 10x mas lento que decoder
    decoder_params = list(model.sam.mask_decoder.parameters())
    decoder_ids    = {id(p) for p in decoder_params}
    encoder_params = [p for p in model.get_trainable_params() if id(p) not in decoder_ids]

    param_groups = [
        {"params": decoder_params, "lr": lr,       "name": "decoder"},
        {"params": encoder_params, "lr": lr * 0.1, "name": "encoder_tail"},
    ]
    # Filtrar grupos vacios
    param_groups = [g for g in param_groups if len(g["params"]) > 0]

    optimizer = AdamW(param_groups, weight_decay=wd)

    num_epochs = config["training"]["num_epochs"]
    sch_cfg    = config.get("scheduler", {})
    scheduler  = ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=sch_cfg.get("factor", 0.5),
        patience=sch_cfg.get("patience", 4),
        min_lr=sch_cfg.get("min_lr", 1e-7),
    )

    criterion = build_loss(config)
    scaler    = GradScaler("cuda", enabled=config["training"]["mixed_precision"])

    # Early stopping (maximizar val_iou)
    patience   = config["early_stopping"]["patience"]
    min_delta  = config["early_stopping"]["min_delta"]
    best_iou   = -1.0
    no_improve = 0
    best_path  = ckpt_dir / "best_model.pth"
    last_path  = ckpt_dir / "last_model.pth"

    history    = {k: [] for k in ["train_loss", "val_loss", "val_iou"]}
    csv_path   = logs_dir / "training_log.csv"
    csv_file   = open(csv_path, "w", newline="", encoding="utf-8")
    writer     = csv.writer(csv_file)
    writer.writerow(["epoch", "train_loss", "val_loss", "val_iou",
                     "f1", "precision", "recall", "lr"])

    print(f"\n[Train] Iniciando  {num_epochs} epochs | "
          f"batch={config['training']['batch_size']} | "
          f"lr={config['training']['learning_rate']:.1e} | "
          f"patience={patience}")
    print("=" * 65)

    total_t0 = time.time()

    for epoch in range(1, num_epochs + 1):
        ep_t0 = time.time()

        train_m = train_one_epoch(
            model, train_loader, optimizer, criterion,
            scaler, device, config, epoch,
        )
        val_m   = validate(model, val_loader, criterion, device, config)
        val_iou = val_m["val_iou"]
        scheduler.step(val_iou)

        elapsed = time.time() - ep_t0
        lr_now  = optimizer.param_groups[0]["lr"]   # LR del decoder (el principal)

        history["train_loss"].append(train_m["train_loss"])
        history["val_loss"].append(val_m["val_loss"])
        history["val_iou"].append(val_iou)

        writer.writerow([epoch,
                         f"{train_m['train_loss']:.5f}",
                         f"{val_m['val_loss']:.5f}",
                         f"{val_iou:.5f}",
                         f"{val_m['f1']:.5f}",
                         f"{val_m['precision']:.5f}",
                         f"{val_m['recall']:.5f}",
                         f"{lr_now:.2e}"])
        csv_file.flush()

        print(
            f"\nEpoch {epoch:3d}/{num_epochs} [{elapsed:.0f}s] | "
            f"Train Loss={train_m['train_loss']:.4f} | "
            f"Val Loss={val_m['val_loss']:.4f} | "
            f"Val IoU={val_iou:.4f} | "
            f"F1={val_m['f1']:.4f} | LR={lr_now:.2e}"
        )

        model.save_checkpoint(last_path, metadata={"epoch": epoch, "val_iou": val_iou})

        if val_iou > best_iou + min_delta:
            best_iou   = val_iou
            no_improve = 0
            model.save_checkpoint(
                best_path,
                metadata={"epoch": epoch, "val_iou": best_iou,
                          "f1": val_m["f1"], "precision": val_m["precision"],
                          "recall": val_m["recall"],
                          "unfreeze_encoder_blocks": n_unfreeze},
            )
            print(f"  --> Mejor modelo guardado (val_iou={best_iou:.4f})")
        else:
            no_improve += 1
            print(f"  No mejora {no_improve}/{patience}")
            if no_improve >= patience:
                print(f"[Train] Early stopping en epoch {epoch}.")
                break

        print()

    csv_file.close()

    total_elapsed = time.time() - total_t0
    print(f"\n{'='*65}")
    print(f"[Train] Completado en {total_elapsed/60:.1f} min")
    print(f"[Train] Mejor val_IoU: {best_iou:.4f}")
    print(f"[Train] Mejor checkpoint: {best_path}")

    save_curves(history, logs_dir / "training_curves.png")
    print(f"[Train] Curvas guardadas: {logs_dir / 'training_curves.png'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fine-tuning SAM para segmentación de copas de árboles"
    )
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    os.chdir(Path(__file__).parent)
    train(args.config)
