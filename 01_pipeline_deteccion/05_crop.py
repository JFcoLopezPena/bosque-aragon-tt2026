"""
Crop individual tree patches from tiles based on filtered detection bboxes.

For each detection the bounding box (+ padding) is used to extract:
  - A uint8 RGB PNG
  - A float32 multiband .npy (5 bands: R, G, B, NIR, NDVI)

Outputs
-------
output/crops/rgb/{mask_id}_{tile_id}.png
output/crops/multiband/{mask_id}_{tile_id}.npy
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import yaml
from PIL import Image
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def _setup_logger(log_dir: Path) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"05_crop_{datetime.now():%Y%m%d_%H%M%S}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    return logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_config(config_path: Path) -> dict:
    with open(config_path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _assert_exists(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"[{label}] not found: {path}")


def _clamp(value: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, value))


# ---------------------------------------------------------------------------
# Core crop
# ---------------------------------------------------------------------------

def _crop_rgb(
    rgb_tile_path: Path,
    bbox_xywh: list[int],
    padding: int,
    out_path: Path,
) -> bool:
    """Crop an RGB PNG tile and save with padding around bbox."""
    img = cv2.imread(str(rgb_tile_path))
    if img is None:
        return False
    h, w = img.shape[:2]
    x, y, bw, bh = bbox_xywh
    x1 = _clamp(x - padding, 0, w - 1)
    y1 = _clamp(y - padding, 0, h - 1)
    x2 = _clamp(x + bw + padding, 0, w)
    y2 = _clamp(y + bh + padding, 0, h)
    crop = img[y1:y2, x1:x2]
    cv2.imwrite(str(out_path), crop)
    return True


def _crop_multiband(
    mb_tile_path: Path,
    bbox_xywh: list[int],
    padding: int,
    out_path: Path,
) -> bool:
    """Crop a multiband .npy tile and save with padding around bbox."""
    arr = np.load(str(mb_tile_path))     # (5, H, W)
    _, h, w = arr.shape
    x, y, bw, bh = bbox_xywh
    x1 = _clamp(x - padding, 0, w - 1)
    y1 = _clamp(y - padding, 0, h - 1)
    x2 = _clamp(x + bw + padding, 0, w)
    y2 = _clamp(y + bh + padding, 0, h)
    crop = arr[:, y1:y2, x1:x2]         # (5, crop_h, crop_w)
    np.save(str(out_path), crop)
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_crop(cfg: dict, log: logging.Logger) -> None:
    output_dir = Path(cfg["paths"]["output_dir"])
    padding    = int(cfg.get("crop_padding", 10))

    detections_path = output_dir / "masks" / "valid_detections.json"
    _assert_exists(detections_path, "valid_detections.json")

    with open(detections_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    detections: list[dict] = data["detections"]

    if not detections:
        log.info("No detections found. Nothing to crop.")
        return

    rgb_tiles_dir = output_dir / "tiles" / "rgb"
    mb_tiles_dir  = output_dir / "tiles" / "multiband"
    rgb_crops_dir = output_dir / "crops" / "rgb"
    mb_crops_dir  = output_dir / "crops" / "multiband"
    rgb_crops_dir.mkdir(parents=True, exist_ok=True)
    mb_crops_dir.mkdir(parents=True, exist_ok=True)

    log.info("Cropping %d detections (padding=%d px) …", len(detections), padding)

    ok_count = 0
    fail_count = 0

    for det in tqdm(detections, desc="Cropping", unit="tree"):
        mask_id    = det["mask_id"]
        tile_id    = det["tile_id"]
        bbox_xywh  = det["bbox_tile_xywh"]

        rgb_tile   = rgb_tiles_dir / f"{tile_id}.png"
        mb_tile    = mb_tiles_dir  / f"{tile_id}.npy"

        stem       = f"{mask_id}_{tile_id}"
        rgb_out    = rgb_crops_dir / f"{stem}.png"
        mb_out     = mb_crops_dir  / f"{stem}.npy"

        # Skip if already done
        if rgb_out.exists() and mb_out.exists():
            ok_count += 1
            continue

        if not rgb_tile.exists():
            log.warning("RGB tile missing: %s", rgb_tile)
            fail_count += 1
            continue
        if not mb_tile.exists():
            log.warning("Multiband tile missing: %s", mb_tile)
            fail_count += 1
            continue

        try:
            rgb_ok = _crop_rgb(rgb_tile, bbox_xywh, padding, rgb_out)
            mb_ok  = _crop_multiband(mb_tile, bbox_xywh, padding, mb_out)
            if rgb_ok and mb_ok:
                ok_count += 1
            else:
                fail_count += 1
                log.error("Crop failed for %s", stem)
        except Exception as exc:
            fail_count += 1
            log.error("Error cropping %s: %s", stem, exc, exc_info=True)

    log.info("Crops done: ok=%d  failed=%d", ok_count, fail_count)
    log.info("RGB  crops → %s", rgb_crops_dir)
    log.info("Multiband  → %s", mb_crops_dir)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    config_path = Path(__file__).parent / "config.yaml"
    cfg = _load_config(config_path)
    output_dir = Path(cfg["paths"]["output_dir"])
    log = _setup_logger(output_dir / "logs")

    log.info("=" * 60)
    log.info("05_crop.py — individual tree crops")
    log.info("=" * 60)

    run_crop(cfg, log)
    log.info("05_crop.py complete.")


if __name__ == "__main__":
    main()
