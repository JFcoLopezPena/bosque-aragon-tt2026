"""
Run SAM (Segment Anything Model) over every RGB tile and save raw masks.

Outputs
-------
output/sam_raw/{tile_id}.pkl   list[dict] with SAM mask dicts per tile
  Each dict keys: segmentation, area, bbox, predicted_iou, stability_score

Checkpoint recovery: tiles with an existing .pkl in sam_raw/ are skipped.
"""

from __future__ import annotations

import logging
import pickle
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def _setup_logger(log_dir: Path) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"03_sam_{datetime.now():%Y%m%d_%H%M%S}.log"
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


def _load_sam(checkpoint: Path, model_type: str, device: str):
    """Load SAM model and return a configured SamAutomaticMaskGenerator."""
    try:
        from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
    except ImportError:
        raise ImportError(
            "segment-anything not found. Run 00_install.py first."
        )

    _assert_exists(checkpoint, "SAM checkpoint")
    model = sam_model_registry[model_type](checkpoint=str(checkpoint))
    model.to(device=device)
    model.eval()
    return model


def _build_mask_generator(model, cfg: dict):
    from segment_anything import SamAutomaticMaskGenerator

    s = cfg["sam"]
    return SamAutomaticMaskGenerator(
        model=model,
        points_per_side=int(s["points_per_side"]),
        pred_iou_thresh=float(s["pred_iou_thresh"]),
        stability_score_thresh=float(s["stability_score_thresh"]),
        min_mask_region_area=int(s["min_mask_area"]),
    )


def _process_tile(
    tile_path: Path,
    mask_generator,
    min_area: int,
    max_area: int,
) -> list[dict]:
    """Run SAM on one tile and return filtered mask list."""
    img_bgr = cv2.imread(str(tile_path))
    if img_bgr is None:
        raise ValueError(f"Could not read image: {tile_path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)   # SAM expects RGB uint8

    masks = mask_generator.generate(img_rgb)

    # Keep only fields needed downstream (saves ~60 % pickle size)
    kept = []
    for m in masks:
        area = int(m["area"])
        if area < min_area or area > max_area:
            continue
        kept.append({
            "segmentation":      m["segmentation"],          # (H, W) bool
            "area":              area,
            "bbox":              m["bbox"],                   # [x, y, w, h]
            "predicted_iou":     float(m["predicted_iou"]),
            "stability_score":   float(m["stability_score"]),
        })
    return kept


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_inference(cfg: dict, log: logging.Logger) -> None:
    output_dir = Path(cfg["paths"]["output_dir"])

    checkpoint  = Path(cfg["paths"]["sam_checkpoint"])
    model_type  = cfg["sam"]["model_type"]
    device      = cfg["sam"]["device"]
    min_area    = int(cfg["sam"]["min_mask_area"])
    max_area    = int(cfg["sam"]["max_mask_area"])

    _assert_exists(checkpoint, "SAM checkpoint")

    tiles_dir   = output_dir / "tiles" / "rgb"
    raw_dir     = output_dir / "sam_raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    tile_paths = sorted(tiles_dir.glob("tile_*.png"))
    if not tile_paths:
        raise RuntimeError(f"No tiles found in {tiles_dir}. Run 02_tiling.py first.")

    # Checkpoint recovery: skip already-processed tiles
    done = {p.stem for p in raw_dir.glob("*.pkl")}
    todo = [p for p in tile_paths if p.stem not in done]
    log.info(
        "Tiles: total=%d  done=%d  to_process=%d",
        len(tile_paths), len(done), len(todo),
    )

    if not todo:
        log.info("All tiles already processed.")
        return

    # Load model
    log.info("Loading SAM %s on %s …", model_type, device)
    model = _load_sam(checkpoint, model_type, device)
    mask_gen = _build_mask_generator(model, cfg)
    log.info("SAM loaded.")

    total_masks = 0
    t0_all = time.perf_counter()

    with tqdm(todo, desc="SAM inference", unit="tile") as pbar:
        for tile_path in pbar:
            t0 = time.perf_counter()
            pkl_path = raw_dir / f"{tile_path.stem}.pkl"

            try:
                masks = _process_tile(tile_path, mask_gen, min_area, max_area)
                with open(pkl_path, "wb") as fh:
                    pickle.dump(masks, fh, protocol=pickle.HIGHEST_PROTOCOL)
                total_masks += len(masks)

                elapsed = time.perf_counter() - t0
                processed = pbar.n + 1
                remaining = len(todo) - processed
                eta_s = elapsed * remaining
                eta_min = eta_s / 60

                pbar.set_postfix({
                    "masks": len(masks),
                    "total": total_masks,
                    "ETA_min": f"{eta_min:.1f}",
                })

            except Exception as exc:
                log.error("Tile %s failed: %s", tile_path.stem, exc, exc_info=True)
                continue
            finally:
                # Free GPU memory between tiles
                if device == "cuda":
                    torch.cuda.empty_cache()

    elapsed_total = time.perf_counter() - t0_all
    log.info(
        "\nInference complete: %d tiles processed, %d masks total, %.1f s",
        len(todo), total_masks, elapsed_total,
    )


def main() -> None:
    config_path = Path(__file__).parent / "config.yaml"
    cfg = _load_config(config_path)
    output_dir = Path(cfg["paths"]["output_dir"])
    log = _setup_logger(output_dir / "logs")

    log.info("=" * 60)
    log.info("03_sam_inference.py — SAM mask generation")
    log.info("=" * 60)

    run_inference(cfg, log)
    log.info("03_sam_inference.py complete.")


if __name__ == "__main__":
    main()
