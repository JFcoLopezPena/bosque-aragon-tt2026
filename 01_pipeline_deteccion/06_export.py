"""
Export all valid detections to a GeoJSON with geographic coordinates (EPSG:32614).

Each feature polygon is derived from the SAM mask contour converted to world
coordinates using the tile affine transform stored in tiles/metadata.json.

Outputs
-------
output/metadata/detections.geojson
"""

from __future__ import annotations

import json
import logging
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import yaml
from affine import Affine
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def _setup_logger(log_dir: Path) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"06_export_{datetime.now():%Y%m%d_%H%M%S}.log"
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


def _px_contour_to_geo(
    contour: np.ndarray,
    tile_transform: Affine,
    simplify_tolerance: float = 0.5,
) -> list[list[float]]:
    """Convert a pixel-space contour to geographic coordinate ring.

    Parameters
    ----------
    contour:
        OpenCV contour array, shape (N, 1, 2) or (N, 2).
    tile_transform:
        Affine transform for the tile (top-left pixel = world origin).
    simplify_tolerance:
        Douglas-Peucker epsilon in pixels to reduce vertex count.
    """
    pts = contour.reshape(-1, 2).astype(float)

    # Simplify contour in pixel space
    if simplify_tolerance > 0 and len(pts) > 6:
        c_approx = cv2.approxPolyDP(
            contour, epsilon=simplify_tolerance, closed=True
        )
        pts = c_approx.reshape(-1, 2).astype(float)

    ring: list[list[float]] = []
    for px_col, px_row in pts:
        geo_x, geo_y = tile_transform * (px_col, px_row)
        ring.append([round(geo_x, 4), round(geo_y, 4)])

    # Close the ring
    if ring and ring[0] != ring[-1]:
        ring.append(ring[0])

    return ring


def _mask_to_polygon(
    mask_path: Path,
    tile_transform: Affine,
) -> list[list[float]] | None:
    """Load mask PNG and return the largest contour as a closed geo ring."""
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    ring = _px_contour_to_geo(largest, tile_transform)
    return ring if len(ring) >= 4 else None


def _print_statistics(
    detections: list[dict],
    log: logging.Logger,
) -> None:
    if not detections:
        log.info("No detections to summarise.")
        return

    areas   = np.array([d["area_m2"]   for d in detections])
    ndvis   = np.array([d["ndvi_mean"] for d in detections])
    by_tile: dict[str, int] = defaultdict(int)
    for d in detections:
        by_tile[d["tile_id"]] += 1

    log.info("\n=== Final statistics ===")
    log.info("  Total trees detected : %d", len(detections))
    log.info("  Area [m²]  mean=%.2f  median=%.2f  min=%.2f  max=%.2f",
             areas.mean(), float(np.median(areas)), areas.min(), areas.max())
    log.info("  NDVI       mean=%.3f  std=%.3f  min=%.3f  max=%.3f",
             ndvis.mean(), ndvis.std(), ndvis.min(), ndvis.max())

    # Simple zone density map (3×3 grid of tile counts)
    tile_ids = list(by_tile.keys())
    if tile_ids:
        rows = [int(t.split("_")[1]) for t in tile_ids]
        cols = [int(t.split("_")[2]) for t in tile_ids]
        log.info(
            "  Tile range: rows %d–%d  cols %d–%d  "
            "(%d tiles with ≥1 detection)",
            min(rows), max(rows), min(cols), max(cols), len(by_tile),
        )

    log.info("========================\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_export(cfg: dict, log: logging.Logger) -> None:
    output_dir = Path(cfg["paths"]["output_dir"])

    detections_path = output_dir / "masks" / "valid_detections.json"
    _assert_exists(detections_path, "valid_detections.json")

    meta_path = output_dir / "tiles" / "metadata.json"
    _assert_exists(meta_path, "tiles/metadata.json")

    masks_dir = output_dir / "masks"
    meta_out  = output_dir / "metadata"
    meta_out.mkdir(parents=True, exist_ok=True)

    with open(detections_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    detections: list[dict] = data["detections"]

    tile_meta = _load_tile_metadata(meta_path)

    log.info("Building GeoJSON for %d detections …", len(detections))

    features: list[dict] = []
    failed = 0

    for det in tqdm(detections, desc="Exporting", unit="tree"):
        tile_id  = det["tile_id"]
        mask_id  = det["mask_id"]
        mask_file = masks_dir / det["mask_file"]

        if tile_id not in tile_meta:
            log.warning("Tile %s not in metadata, skipping %s.", tile_id, mask_id)
            failed += 1
            continue

        # Reconstruct affine transform from stored coefficients (a,b,c,d,e,f)
        tf_coeffs  = tile_meta[tile_id]["transform"]
        a, b, c, d, e, f = tf_coeffs
        tile_transform = Affine(a, b, c, d, e, f)

        ring = _mask_to_polygon(mask_file, tile_transform) if mask_file.exists() else None

        if ring is None:
            # Fall back to bbox polygon
            col_off = int(tile_meta[tile_id]["col_off"])
            row_off = int(tile_meta[tile_id]["row_off"])
            x, y, w, h = det["bbox_tile_xywh"]
            corners = [
                (col_off + x,     row_off + y),
                (col_off + x + w, row_off + y),
                (col_off + x + w, row_off + y + h),
                (col_off + x,     row_off + y + h),
                (col_off + x,     row_off + y),
            ]
            # Use global transform from first tile (rough approximation)
            # For accurate coordinates, we want the RGB raster transform
            # which equals the first tile transform shifted by (0,0).
            ring = [[round(tile_transform.c + px * tile_transform.a, 4),
                     round(tile_transform.f + py * tile_transform.e, 4)]
                    for px, py in corners]

        feature = {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [ring],
            },
            "properties": {
                "id":              mask_id,
                "tile_id":         tile_id,
                "area_m2":         det["area_m2"],
                "ndvi_mean":       det["ndvi_mean"],
                "ndvi_std":        det["ndvi_std"],
                "circularity":     det["circularity"],
                "aspect_ratio":    det["aspect_ratio"],
                "score_sam":       det["sam_iou_score"],
                "sam_stability":   det["sam_stability"],
                "mask_file":       det["mask_file"],
            },
        }
        features.append(feature)

    geojson = {
        "type": "FeatureCollection",
        "crs": {
            "type": "name",
            "properties": {"name": "urn:ogc:def:crs:EPSG::32614"},
        },
        "features": features,
    }

    geojson_path = meta_out / "detections.geojson"
    with open(geojson_path, "w", encoding="utf-8") as fh:
        json.dump(geojson, fh, indent=2)

    log.info("GeoJSON → %s  (%d features, %d failed)", geojson_path, len(features), failed)
    _print_statistics(detections, log)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    config_path = Path(__file__).parent / "config.yaml"
    cfg = _load_config(config_path)
    output_dir = Path(cfg["paths"]["output_dir"])
    log = _setup_logger(output_dir / "logs")

    log.info("=" * 60)
    log.info("06_export.py — GeoJSON export")
    log.info("=" * 60)

    run_export(cfg, log)
    log.info("06_export.py complete.")


if __name__ == "__main__":
    main()
