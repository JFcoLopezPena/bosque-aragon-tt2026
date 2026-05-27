"""
Inferencia con el modelo fine-tuneado sobre imágenes nuevas.
Acepta una imagen individual o un directorio.
Genera: máscara binaria .png + overlay sobre la imagen original.

Uso:
    python predict.py --input ruta/imagen.png
    python predict.py --input ruta/directorio/
    python predict.py --input ruta/ --threshold 0.4
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import autocast
import yaml

sys.path.insert(0, str(Path(__file__).parent))

from model import SegmentationSAM

IMG_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def load_config(path: str | Path = "config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def preprocess_image(
    img_bgr: np.ndarray,
    target_size: int = 512,
) -> tuple[torch.Tensor, tuple[int, int]]:
    """
    Preprocesa imagen BGR→RGB, redimensiona a target_size y convierte a tensor [1, 3, H, W] ∈ [0,1].
    Devuelve tensor y tamaño original (h, w).
    """
    orig_h, orig_w = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
    tensor = torch.from_numpy(img_resized.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
    return tensor, (orig_h, orig_w)


def postprocess_mask(
    logits:    torch.Tensor,   # [1, 1, H, W]
    orig_size: tuple[int, int],
    threshold: float = 0.5,
) -> np.ndarray:
    """Aplica sigmoid, umbraliza y devuelve máscara uint8 [H_orig, W_orig] {0, 255}."""
    prob = torch.sigmoid(logits[0, 0]).cpu().numpy()
    mask = (prob > threshold).astype(np.uint8)
    mask_resized = cv2.resize(mask, (orig_size[1], orig_size[0]), interpolation=cv2.INTER_NEAREST)
    return mask_resized * 255


def make_overlay(img_bgr: np.ndarray, mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Superpone la máscara binaria (255=copa) sobre la imagen con color verde."""
    overlay  = img_bgr.copy()
    copa_px  = mask > 127
    green    = np.array([0, 180, 0], dtype=np.uint8)
    overlay[copa_px] = (
        overlay[copa_px].astype(np.float32) * (1 - alpha) + green.astype(np.float32) * alpha
    ).astype(np.uint8)
    return overlay


@torch.no_grad()
def predict_single(
    model:       SegmentationSAM,
    img_path:    Path,
    out_dir:     Path,
    device:      torch.device,
    config:      dict,
    threshold:   float = 0.5,
    use_amp:     bool  = True,
) -> None:
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        print(f"  WARN: no se pudo leer {img_path.name}, saltando")
        return

    dataset_size = config["training"]["image_size"]
    tensor, orig_size = preprocess_image(img_bgr, target_size=dataset_size)

    # Upscale a 1024 para SAM
    tensor_1024 = F.interpolate(tensor, size=(1024, 1024), mode="bilinear", align_corners=False)
    tensor_1024 = tensor_1024.to(device)

    with autocast("cuda", enabled=use_amp):
        logits = model(tensor_1024, output_size=(dataset_size, dataset_size))

    mask    = postprocess_mask(logits, orig_size, threshold=threshold)
    overlay = make_overlay(img_bgr, mask)

    stem = img_path.stem
    cv2.imwrite(str(out_dir / f"{stem}_mask.png"),    mask)
    cv2.imwrite(str(out_dir / f"{stem}_overlay.png"), overlay)


def predict(
    input_path:  str,
    config_path: str = "config.yaml",
    threshold:   float = 0.5,
    out_dir:     str | None = None,
) -> None:
    config = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = config["training"]["mixed_precision"]

    ckpt_dir  = Path(config["paths"]["checkpoint_dir"])
    best_path = ckpt_dir / "best_model.pth"
    if not best_path.exists():
        raise FileNotFoundError(f"Checkpoint no encontrado: {best_path}")

    print(f"[Predict] Cargando modelo desde {best_path} ...")
    model = SegmentationSAM.load_checkpoint(
        ft_checkpoint=best_path,
        sam_checkpoint=config["model"]["checkpoint_path"],
    ).to(device)
    model.eval()

    input_path = Path(input_path)
    if input_path.is_file():
        img_paths = [input_path]
    elif input_path.is_dir():
        img_paths = sorted(p for p in input_path.iterdir() if p.suffix.lower() in IMG_EXTS)
    else:
        raise FileNotFoundError(f"Ruta no encontrada: {input_path}")

    if not img_paths:
        print("[Predict] No se encontraron imágenes.")
        return

    results_dir = Path(out_dir) if out_dir else Path(config["paths"]["results_dir"]) / "predictions"
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Predict] {len(img_paths)} imágenes → {results_dir}")

    for i, img_path in enumerate(img_paths, 1):
        predict_single(model, img_path, results_dir, device, config, threshold, use_amp)
        if i % 10 == 0 or i == len(img_paths):
            print(f"  {i}/{len(img_paths)} procesadas")

    print(f"[Predict] Listo. Resultados en {results_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Predicción de copas de árboles con SAM fine-tuneado"
    )
    parser.add_argument("--input",     required=True,    help="Imagen o directorio de imágenes")
    parser.add_argument("--config",    default="config.yaml")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Umbral de sigmoid para binarizar (default: 0.5)")
    parser.add_argument("--out_dir",   default=None,
                        help="Directorio de salida (default: results/predictions/)")
    args = parser.parse_args()

    os.chdir(Path(__file__).parent)
    predict(args.input, args.config, args.threshold, args.out_dir)
