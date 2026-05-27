"""
Filter SAM masks by area, NDVI, circularity and aspect ratio.
Apply NMS across overlapping tiles to remove duplicate detections.

Outputs
-------
output/masks/{mask_id}.png               binary mask in tile pixel space (uint8)
output/masks/valid_detections.json       metadata for every surviving detection
"""

from __future__ import annotations

import json
import logging
import math
import pickle
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def _setup_logger(log_dir: Path) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"04_filter_{datetime.now():%Y%m%d_%H%M%S}.log"
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


def _load_tile_metadata(meta_path: Path) -> dict[str, dict]:
    with open(meta_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    return {d["tile_id"]: d for d in data["tiles"]}


# ---------------------------------------------------------------------------
# Geometry metrics
# ---------------------------------------------------------------------------

def _largest_contour(mask: np.ndarray) -> np.ndarray | None:
    """Return largest contour from a binary mask, or None if mask is empty."""
    mask_u8 = (mask > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


def _circularity(contour: np.ndarray) -> float:
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, closed=True)
    if perimeter < 1e-6:
        return 0.0
    return 4 * math.pi * area / (perimeter ** 2)


def _aspect_ratio(bbox_xywh: list[int]) -> float:
    _, _, w, h = bbox_xywh
    if h == 0:
        return float("inf")
    return max(w, h) / max(min(w, h), 1)


def _ndvi_stats(
    mask: np.ndarray,
    multiband: np.ndarray,
) -> tuple[float, float]:
    """Return (ndvi_mean, ndvi_std) inside the mask using band index 4 (NDVI)."""
    ndvi_band = multiband[4]                 # (H, W)
    pixels    = ndvi_band[mask > 0]
    if pixels.size == 0:
        return 0.0, 0.0
    return float(pixels.mean()), float(pixels.std())


# ---------------------------------------------------------------------------
# NMS
# ---------------------------------------------------------------------------

def _bbox_to_xyxy(col_off: int, row_off: int, bbox_xywh: list[int]) -> list[int]:
    """Convert tile-local [x, y, w, h] bbox to global [x1, y1, x2, y2]."""
    x, y, w, h = bbox_xywh
    return [col_off + x, row_off + y, col_off + x + w, row_off + y + h]


def _iou_xyxy(a: list[int], b: list[int]) -> float:
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _nms(detections: list[dict], iou_thresh: float) -> list[dict]:
    """Greedy NMS: sort by SAM predicted_iou, suppress overlapping boxes."""
    if not detections:
        return []
    sorted_dets = sorted(detections, key=lambda d: d["sam_iou_score"], reverse=True)
    kept: list[dict] = []
    suppressed: set[int] = set()
    for i, det_i in enumerate(sorted_dets):
        if i in suppressed:
            continue
        kept.append(det_i)
        box_i = det_i["bbox_global_xyxy"]
        for j in range(i + 1, len(sorted_dets)):
            if j in suppressed:
                continue
            if _iou_xyxy(box_i, sorted_dets[j]["bbox_global_xyxy"]) > iou_thresh:
                suppressed.add(j)
    return kept


# ---------------------------------------------------------------------------
# Main filter loop
# ---------------------------------------------------------------------------

def run_filter(cfg: dict, log: logging.Logger) -> None:
    output_dir    = Path(cfg["paths"]["output_dir"])
    resolution    = float(cfg["resolution"])         # m/px
    px_to_m2      = resolution ** 2

    sam_cfg        = cfg["sam"]
    flt            = cfg["filter"]
    min_area_px    = int(sam_cfg["min_mask_area"])
    max_area_px    = int(sam_cfg["max_mask_area"])
    ndvi_min       = float(flt["ndvi_min"])
    circ_min       = float(flt["circularity_min"])
    ar_max         = float(flt["aspect_ratio_max"])
    nms_iou        = float(flt["nms_iou_threshold"])

    raw_dir        = output_dir / "sam_raw"
    mb_dir         = output_dir / "tiles" / "multiband"
    masks_dir      = output_dir / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)

    meta_path = output_dir / "tiles" / "metadata.json"
    _assert_exists(meta_path, "tiles/metadata.json")
    tile_meta = _load_tile_metadata(meta_path)

    pkl_files = sorted(raw_dir.glob("*.pkl"))
    if not pkl_files:
        raise RuntimeError(f"No .pkl files in {raw_dir}. Run 03_sam_inference.py first.")

    log.info("Processing %d tile .pkl files …", len(pkl_files))

    counts = {"total": 0, "after_size": 0, "after_ndvi": 0, "after_shape": 0}
    all_detections: list[dict] = []

    for pkl_path in tqdm(pkl_files, desc="Filtering", unit="tile"):
        tile_id = pkl_path.stem
        if tile_id not in tile_meta:
            log.warning("Tile %s not in metadata, skipping.", tile_id)
            continue

        meta    = tile_meta[tile_id]
        col_off = int(meta["col_off"])
        row_off = int(meta["row_off"])

        # Load multiband tile
        mb_path = mb_dir / f"{tile_id}.npy"
        if not mb_path.exists():
            log.warning("Multiband file missing for %s, skipping.", tile_id)
            continue
        multiband = np.load(str(mb_path))    # (5, H, W) float32

        with open(pkl_path, "rb") as fh:
            raw_masks: list[dict] = pickle.load(fh)

        counts["total"] += len(raw_masks)

        for mask_dict in raw_masks:
            mask: np.ndarray = mask_dict["segmentation"].astype(np.uint8)
            area_px: int     = int(mask_dict["area"])
            bbox_xywh        = list(map(int, mask_dict["bbox"]))

            # --- Size filter (already pre-filtered in step 03 but double-check)
            if not (min_area_px <= area_px <= max_area_px):
                continue
            counts["after_size"] += 1

            # --- NDVI filter
            # Clamp mask to multiband tile dimensions
            mh, mw = multiband.shape[1], multiband.shape[2]
            if mask.shape[0] != mh or mask.shape[1] != mw:
                log.debug("Mask/multiband shape mismatch in %s, resizing.", tile_id)
                mask = cv2.resize(mask, (mw, mh), interpolation=cv2.INTER_NEAREST)

            ndvi_mean, ndvi_std = _ndvi_stats(mask, multiband)
            if ndvi_mean < ndvi_min:
                continue
            counts["after_ndvi"] += 1

            # --- Shape filter
            contour = _largest_contour(mask)
            if contour is None:
                continue
            circ = _circularity(contour)
            ar   = _aspect_ratio(bbox_xywh)

            if circ < circ_min or ar > ar_max:
                continue
            counts["after_shape"] += 1

            # --- Accumulate detection
            bbox_global_xyxy = _bbox_to_xyxy(col_off, row_off, bbox_xywh)
            all_detections.append({
                "_tile_id":         tile_id,
                "_mask":            mask,            # kept temporarily for NMS + save
                "tile_id":          tile_id,
                "bbox_tile_xywh":   bbox_xywh,
                "bbox_global_xyxy": bbox_global_xyxy,
                "area_px":          area_px,
                "area_m2":          round(area_px * px_to_m2, 4),
                "ndvi_mean":        round(ndvi_mean, 4),
                "ndvi_std":         round(ndvi_std, 4),
                "circularity":      round(circ, 4),
                "aspect_ratio":     round(ar, 4),
                "sam_iou_score":    round(float(mask_dict["predicted_iou"]), 4),
                "sam_stability":    round(float(mask_dict["stability_score"]), 4),
            })

    log.info(
        "Filter counts:  total=%d  →  size=%d  →  ndvi=%d  →  shape=%d",
        counts["total"], counts["after_size"], counts["after_ndvi"], counts["after_shape"],
    )

    # ------------------------------------------------------------------
    # NMS across overlapping tiles
    # ------------------------------------------------------------------
    log.info("Running NMS (iou_thresh=%.2f) on %d detections …", nms_iou, len(all_detections))
    kept = _nms(all_detections, nms_iou)
    log.info("After NMS: %d detections", len(kept))

    # ------------------------------------------------------------------
    # Save masks + build final JSON
    # ------------------------------------------------------------------
    final_detections: list[dict] = []

    for det_id, det in enumerate(
        tqdm(kept, desc="Saving masks", unit="mask")
    ):
        tile_id  = det["tile_id"]
        mask_id  = f"arbol_{det_id:06d}"
        filename = f"{mask_id}_{tile_id}.png"
        mask_path = masks_dir / filename

        try:
            mask_img = (det["_mask"] > 0).astype(np.uint8) * 255
            cv2.imwrite(str(mask_path), mask_img)
        except Exception as exc:
            log.error("Could not save mask %s: %s", mask_path, exc)
            continue

        record = {k: v for k, v in det.items() if not k.startswith("_")}
        record["mask_id"]   = mask_id
        record["mask_file"] = filename
        final_detections.append(record)

    detections_json = masks_dir / "valid_detections.json"
    with open(detections_json, "w", encoding="utf-8") as fh:
        json.dump({"detections": final_detections}, fh, indent=2)

    log.info("Saved %d mask PNGs to %s", len(final_detections), masks_dir)
    log.info("Detection metadata → %s", detections_json)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    config_path = Path(__file__).parent / "config.yaml"
    cfg = _load_config(config_path)
    output_dir = Path(cfg["paths"]["output_dir"])
    log = _setup_logger(output_dir / "logs")

    log.info("=" * 60)
    log.info("04_filter.py — mask filtering + NMS")
    log.info("=" * 60)

    run_filter(cfg, log)
    log.info("04_filter.py complete.")


if __name__ == "__main__":
    main()
