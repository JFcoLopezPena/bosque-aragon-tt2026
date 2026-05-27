"""
Fine-tuning de SAM ViT-B para clasificacion fitosanitaria (Ronda 2).

Early stopping: maximiza val_F1_enfermo.
LR diferencial: cls_head x2 | decoder x1 | encoder_tail x0.1

Uso:
    python train.py
    python train.py --config path/to/config.yaml
    python train.py --resume                        # continua desde best_model.pth
"""
from __future__ import annotations

import argparse
import csv
import gc
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
from torch.optim.lr_scheduler import CosineAnnealingLR
import yaml

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))

from dataset import create_dataloaders
from losses  import CombinedLoss
from model   import ClassificationSAM

# Optimizaciones globales de rendimiento
torch.backends.cudnn.benchmark         = True   # cuDNN busca el kernel mas rapido
torch.backends.cuda.matmul.allow_tf32  = True   # TF32 en multiplicaciones de matrices


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


def print_vram() -> None:
    if torch.cuda.is_available():
        free_b, total_b = torch.cuda.mem_get_info()
        print(f"[VRAM] {free_b/1024**3:.1f} GB libre de {total_b/1024**3:.1f} GB total")


def compute_cls_metrics(
    all_logits: torch.Tensor,   # [N, 2]
    all_labels: torch.Tensor,   # [N] int64
) -> Dict[str, float]:
    preds  = all_logits.argmax(dim=1)
    acc    = (preds == all_labels).float().mean().item()
    total  = len(all_labels)
    counts = {c: (all_labels == c).sum().item() for c in (0, 1)}

    metrics: Dict[str, float] = {"accuracy": acc}
    f1s = {}

    for cls_idx, cls_name in ((0, "sano"), (1, "enfermo")):
        tp = ((preds == cls_idx) & (all_labels == cls_idx)).sum().float()
        fp = ((preds == cls_idx) & (all_labels != cls_idx)).sum().float()
        fn = ((preds != cls_idx) & (all_labels == cls_idx)).sum().float()

        prec = (tp / (tp + fp + 1e-8)).item()
        rec  = (tp / (tp + fn + 1e-8)).item()
        f1   = 2 * prec * rec / (prec + rec + 1e-8)

        metrics[f"precision_{cls_name}"] = prec
        metrics[f"recall_{cls_name}"]    = rec
        metrics[f"f1_{cls_name}"]        = f1
        f1s[cls_idx] = f1

    metrics["f1_macro"]    = (f1s[0] + f1s[1]) / 2
    metrics["f1_weighted"] = (f1s[0] * counts[0] + f1s[1] * counts[1]) / max(total, 1)
    return metrics


def compute_seg_iou(
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


def save_curves(history: dict, path: Path) -> None:
    epochs = list(range(1, len(history["train_loss"]) + 1))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(epochs, history["train_loss"], label="train")
    axes[0].plot(epochs, history["val_loss"],   label="val")
    axes[0].set_title("Loss"); axes[0].legend(); axes[0].set_xlabel("Epoch")

    axes[1].plot(epochs, history["val_f1_sano"],    label="F1 SANO")
    axes[1].plot(epochs, history["val_f1_enfermo"], label="F1 ENFERMO")
    axes[1].plot(epochs, history["val_f1_macro"],   label="F1 Macro", linestyle="--")
    axes[1].set_title("Val F1 por clase"); axes[1].legend()
    axes[1].set_xlabel("Epoch"); axes[1].set_ylim(0, 1)

    axes[2].plot(epochs, history["val_iou_seg"], color="tab:green")
    axes[2].axhline(y=0.615, color="red", linestyle="--", label="R1 baseline (0.615)")
    axes[2].set_title("Val IoU Segmentacion")
    axes[2].set_xlabel("Epoch"); axes[2].set_ylim(0, 1); axes[2].legend()

    plt.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ── Train / Validate ──────────────────────────────────────────────────────────

def train_one_epoch(
    model:     ClassificationSAM,
    loader:    torch.utils.data.DataLoader,
    optimizer: AdamW,
    criterion: CombinedLoss,
    scaler:    GradScaler,
    device:    torch.device,
    config:    dict,
    epoch:     int,
) -> Dict[str, float]:
    model.train()
    total_loss = cls_sum = seg_sum = 0.0
    n          = len(loader)
    log_every  = config["logging"]["log_every_n_batches"]
    use_amp    = config["training"]["mixed_precision"]
    clip_norm  = config["training"]["gradient_clip_max_norm"]
    image_size = config["training"]["image_size"]
    t0         = time.time()

    for i, (images, masks, labels) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        masks  = masks.to( device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        images_1024 = F.interpolate(images, (1024, 1024), mode="bilinear", align_corners=False)

        optimizer.zero_grad(set_to_none=True)

        with autocast("cuda", enabled=use_amp):
            logits_cls, logits_mask = model(images_1024, output_size=(image_size, image_size))
            loss, ld = criterion(logits_cls, logits_mask, labels, masks)

        if loss.requires_grad:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            all_params = [p for g in optimizer.param_groups for p in g["params"]]
            torch.nn.utils.clip_grad_norm_(all_params, clip_norm)
            scaler.step(optimizer)
            scaler.update()

        total_loss += ld["total_loss"]
        cls_sum    += ld["cls_loss"]
        seg_sum    += ld["seg_loss"]

        if (i + 1) % log_every == 0 or i == n - 1:
            elapsed = time.time() - t0
            print(f"  Epoch {epoch:3d} | Batch {i+1:4d}/{n} | "
                  f"Loss={total_loss/(i+1):.4f} "
                  f"Cls={ld['cls_loss']:.4f} Seg={ld['seg_loss']:.4f} | {elapsed:.0f}s")

    return {
        "train_loss": total_loss / n,
        "train_cls":  cls_sum    / n,
        "train_seg":  seg_sum    / n,
    }


@torch.no_grad()
def validate(
    model:     ClassificationSAM,
    loader:    torch.utils.data.DataLoader,
    criterion: CombinedLoss,
    device:    torch.device,
    config:    dict,
) -> Dict[str, float]:
    model.eval()
    total_loss      = 0.0
    all_logits_cls  = []
    all_logits_mask = []
    all_labels      = []
    all_masks       = []
    use_amp         = config["training"]["mixed_precision"]
    image_size      = config["training"]["image_size"]

    for images, masks, labels in loader:
        images = images.to(device, non_blocking=True)
        masks  = masks.to( device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        images_1024 = F.interpolate(images, (1024, 1024), mode="bilinear", align_corners=False)

        with autocast("cuda", enabled=use_amp):
            logits_cls, logits_mask = model(images_1024, output_size=(image_size, image_size))
            _, ld = criterion(logits_cls, logits_mask, labels, masks)

        total_loss      += ld["total_loss"]
        all_logits_cls.append( logits_cls.cpu())
        all_logits_mask.append(logits_mask.cpu())
        all_labels.append(     labels.cpu())
        all_masks.append(      masks.cpu())

    all_logits_cls  = torch.cat(all_logits_cls)
    all_logits_mask = torch.cat(all_logits_mask)
    all_labels      = torch.cat(all_labels)
    all_masks       = torch.cat(all_masks)

    m                = compute_cls_metrics(all_logits_cls, all_labels)
    m["val_loss"]    = total_loss / len(loader)
    m["val_iou_seg"] = compute_seg_iou(all_logits_mask, all_masks)
    return m


# ── Pipeline principal ────────────────────────────────────────────────────────

def train(config_path: str = "config.yaml", resume: bool = False) -> None:
    config = load_config(config_path)
    set_seed(config["training"]["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Train] Dispositivo: {device}")
    if torch.cuda.is_available():
        print(f"[Train] GPU: {torch.cuda.get_device_name(0)}")
    print_vram()

    ckpt_dir = Path(config["paths"]["checkpoint_dir"])
    logs_dir = Path(config["paths"]["logs_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    print("\n[Train] Creando DataLoaders ...")
    train_loader, val_loader, _ = create_dataloaders(config)

    print("\n[Train] Cargando modelo ...")
    model = ClassificationSAM(
        sam_checkpoint=config["model"]["sam_checkpoint"],
        r1_checkpoint=config["model"]["r1_checkpoint"],
        unfreeze_encoder_blocks=config["model"].get("unfreeze_encoder_blocks", 2),
        num_classes=config["model"]["num_classes"],
    ).to(device)

    cw_list       = config["training"].get("class_weights", [1.0, 1.0])
    class_weights = torch.tensor(cw_list, dtype=torch.float32).to(device)
    print(f"[Train] Class weights: sano={cw_list[0]}, enfermo={cw_list[1]}")

    criterion    = CombinedLoss(config, class_weights=class_weights)
    lr           = config["training"]["learning_rate"]
    wd           = config["training"]["weight_decay"]
    param_groups = [g for g in model.get_param_groups(lr) if g["params"]]
    optimizer    = AdamW(param_groups, weight_decay=wd)

    num_epochs = config["training"]["num_epochs"]
    scheduler  = CosineAnnealingLR(
        optimizer,
        T_max=config["scheduler"]["T_max"],
        eta_min=config["scheduler"]["eta_min"],
    )
    scaler     = GradScaler("cuda", enabled=config["training"]["mixed_precision"])
    patience   = config["training"]["early_stopping_patience"]
    best_path  = ckpt_dir / "best_model.pth"
    last_path  = ckpt_dir / "last_model.pth"

    # ── Resume from checkpoint ────────────────────────────────────────────────
    start_epoch = 1
    best_f1     = -1.0
    no_improve  = 0
    csv_mode    = "w"

    if resume and best_path.exists():
        ckpt_data   = torch.load(str(best_path), map_location="cpu", weights_only=False)
        metadata    = ckpt_data.get("metadata", {})
        start_epoch = metadata.get("epoch", 1) + 1
        best_f1     = metadata.get("f1_enfermo", -1.0)
        model.load_state_dict(ckpt_data["model_state_dict"])
        # Sync scheduler: advance past already-completed epochs
        for _ in range(start_epoch - 1):
            scheduler.step()
        csv_mode = "a"
        print(f"[Train] Resumiendo desde epoch {start_epoch} "
              f"| best_F1_enfermo={best_f1:.4f}")
        print_vram()
    elif resume:
        print(f"[WARN] --resume solicitado pero no existe {best_path}, iniciando desde cero.")

    # ── Historial en memoria (solo epocas de esta ejecucion) ──────────────────
    history = {k: [] for k in [
        "train_loss", "val_loss",
        "val_f1_sano", "val_f1_enfermo", "val_f1_macro", "val_iou_seg",
    ]}

    csv_path = logs_dir / "training_log.csv"
    csv_file = open(csv_path, csv_mode, newline="", encoding="utf-8")
    writer   = csv.writer(csv_file)
    if csv_mode == "w":
        writer.writerow([
            "epoch", "train_loss", "val_loss", "val_acc",
            "val_F1_sano", "val_F1_enfermo", "val_F1_macro", "val_IoU_seg", "lr",
        ])

    n_train = len(train_loader.dataset)
    n_val   = len(val_loader.dataset)
    print(f"\n[Train] Iniciando desde epoch {start_epoch}/{num_epochs} | "
          f"train={n_train} | val={n_val} | "
          f"batch={config['training']['batch_size']} | "
          f"image_size={config['training']['image_size']} | "
          f"lr={lr:.1e} | patience={patience}")
    print("=" * 70)

    total_t0 = time.time()

    for epoch in range(start_epoch, num_epochs + 1):
        # Liberar memoria antes de cada epoca
        gc.collect()
        torch.cuda.empty_cache()

        ep_t0 = time.time()

        train_m    = train_one_epoch(
            model, train_loader, optimizer, criterion,
            scaler, device, config, epoch,
        )
        val_m      = validate(model, val_loader, criterion, device, config)
        f1_enfermo = val_m["f1_enfermo"]
        scheduler.step()

        elapsed = time.time() - ep_t0
        lr_now  = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(    train_m["train_loss"])
        history["val_loss"].append(      val_m["val_loss"])
        history["val_f1_sano"].append(   val_m["f1_sano"])
        history["val_f1_enfermo"].append(f1_enfermo)
        history["val_f1_macro"].append(  val_m["f1_macro"])
        history["val_iou_seg"].append(   val_m["val_iou_seg"])

        writer.writerow([
            epoch,
            f"{train_m['train_loss']:.5f}",
            f"{val_m['val_loss']:.5f}",
            f"{val_m['accuracy']:.5f}",
            f"{val_m['f1_sano']:.5f}",
            f"{f1_enfermo:.5f}",
            f"{val_m['f1_macro']:.5f}",
            f"{val_m['val_iou_seg']:.5f}",
            f"{lr_now:.2e}",
        ])
        csv_file.flush()

        print(
            f"\nEpoch {epoch:3d}/{num_epochs} [{elapsed:.0f}s] | "
            f"Loss={train_m['train_loss']:.4f} | "
            f"Val Loss={val_m['val_loss']:.4f} | "
            f"Acc={val_m['accuracy']:.4f} | "
            f"F1-E={f1_enfermo:.4f} | "
            f"F1-S={val_m['f1_sano']:.4f} | "
            f"IoU={val_m['val_iou_seg']:.4f} | "
            f"LR={lr_now:.2e}"
        )

        model.save_checkpoint(last_path, metadata={"epoch": epoch, "f1_enfermo": f1_enfermo})

        if f1_enfermo > best_f1 + 1e-4:
            best_f1    = f1_enfermo
            no_improve = 0
            model.save_checkpoint(best_path, metadata={
                "epoch":       epoch,
                "f1_enfermo":  best_f1,
                "f1_sano":     val_m["f1_sano"],
                "f1_macro":    val_m["f1_macro"],
                "accuracy":    val_m["accuracy"],
                "val_iou_seg": val_m["val_iou_seg"],
            })
            print(f"  --> Mejor modelo guardado (F1_enfermo={best_f1:.4f})")
        else:
            no_improve += 1
            print(f"  No mejora {no_improve}/{patience}")
            if no_improve >= patience:
                print(f"[Train] Early stopping en epoch {epoch}.")
                break

        print()

    csv_file.close()

    total_elapsed = time.time() - total_t0
    print(f"\n{'='*70}")
    print(f"[Train] Completado en {total_elapsed/60:.1f} min")
    print(f"[Train] Mejor F1_enfermo (val): {best_f1:.4f}")
    if best_f1 >= 0.65:
        print(f"[Train] Objetivo F1_enfermo >= 0.65 -- ALCANZADO")
    else:
        print(f"[Train] Objetivo F1_enfermo >= 0.65 -- no alcanzado (max={best_f1:.4f})")
        print(f"[Train] Sugerencias: aumentar class_weight[1], reducir lr, mas epochs")
    print(f"[Train] Mejor checkpoint: {best_path}")

    save_curves(history, logs_dir / "training_curves.png")
    print(f"[Train] Curvas guardadas: {logs_dir / 'training_curves.png'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fine-tuning SAM para clasificacion fitosanitaria"
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continua desde best_model.pth (mantiene best_f1 y epoch registrados)",
    )
    args = parser.parse_args()
    os.chdir(Path(__file__).parent)
    train(args.config, resume=args.resume)
