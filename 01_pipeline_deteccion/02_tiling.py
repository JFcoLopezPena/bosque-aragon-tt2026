"""
Divide the aligned ortomosaico into 1024×1024 tiles with 128 px overlap.

Outputs
-------
output/tiles/rgb/{tile_id}.png        8-bit RGB for Label Studio / SAM
output/tiles/multiband/{tile_id}.npy  float32 array (5, H, W): R, G, B, NIR, NDVI
output/tiles/metadata.json            per-tile offsets and geo-transform
"""

from __future__ import annotations

import json
import logging
import math
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import rasterio
import yaml
from PIL import Image
from rasterio.windows import Window
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def _setup_logger(log_dir: Path) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"02_tiling_{datetime.now():%Y%m%d_%H%M%S}.log"
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


def _normalise_rgb_to_uint8(arr: np.ndarray) -> np.ndarray:
    """Convert an (H, W) or (C, H, W) uint16/uint8 array to uint8."""
    if arr.dtype == np.uint8:
        return arr
    # uint16 → rescale to [0, 255]
    arr = arr.astype(np.float32)
    arr = (arr / 65535.0 * 255.0).clip(0, 255).astype(np.uint8)
    return arr


def _valid_fraction(rgb_tile: np.ndarray, nodata_val: int | float | None) -> float:
    """Return fraction of valid (non-nodata) pixels in a (C, H, W) array."""
    total = rgb_tile.shape[1] * rgb_tile.shape[2]
    if total == 0:
        return 0.0
    if nodata_val is not None:
        invalid = np.all(rgb_tile == nodata_val, axis=0).sum()
    else:
        # Treat all-zero pixels as nodata
        invalid = np.all(rgb_tile == 0, axis=0).sum()
    return 1.0 - invalid / total


# ---------------------------------------------------------------------------
# Core tiling
# ---------------------------------------------------------------------------

def run_tiling(cfg: dict, log: logging.Logger) -> None:
    output_dir   = Path(cfg["paths"]["output_dir"])
    aligned_dir  = output_dir / "aligned"

    rgb_path = aligned_dir / "rgb_aligned.tif"
    nir_path = aligned_dir / "nir_aligned.tif"
    r_path   = aligned_dir / "r_aligned.tif"

    for p, label in [(rgb_path, "rgb_aligned"), (nir_path, "nir_aligned"), (r_path, "r_aligned")]:
        _assert_exists(p, label)

    tile_size    = int(cfg["tiling"]["tile_size"])
    overlap      = int(cfg["tiling"]["overlap"])
    min_valid    = float(cfg["tiling"]["min_valid_pixels"])
    stride       = tile_size - overlap

    rgb_out_dir  = output_dir / "tiles" / "rgb"
    mb_out_dir   = output_dir / "tiles" / "multiband"
    rgb_out_dir.mkdir(parents=True, exist_ok=True)
    mb_out_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = output_dir / "tiles" / "metadata.json"

    # Load existing metadata to support resume
    existing_meta: dict[str, dict] = {}
    if metadata_path.exists():
        with open(metadata_path, "r", encoding="utf-8") as fh:
            existing_meta = {d["tile_id"]: d for d in json.load(fh).get("tiles", [])}
        log.info("Resuming: %d tiles already in metadata.json", len(existing_meta))

    with (
        rasterio.open(rgb_path) as rgb_src,
        rasterio.open(nir_path) as nir_src,
        rasterio.open(r_path)   as r_src,
    ):
        width      = rgb_src.width
        height     = rgb_src.height
        transform  = rgb_src.transform
        nodata_val = rgb_src.nodata

        n_cols = math.ceil(width  / stride)
        n_rows = math.ceil(height / stride)
        total  = n_cols * n_rows

        log.info(
            "Raster: %d×%d px  |  stride=%d  |  tiles: %d cols × %d rows = %d",
            width, height, stride, n_cols, n_rows, total,
        )

        tiles_meta: list[dict] = list(existing_meta.values())
        skipped_nodata = 0
        skipped_cached = 0
        written = 0

        positions = [
            (ri, ci)
            for ri in range(n_rows)
            for ci in range(n_cols)
        ]

        for ri, ci in tqdm(positions, desc="Tiling", unit="tile"):
            tile_id = f"tile_{ri:04d}_{ci:04d}"

            # --- checkpoint: skip if already done --------------------------
            if tile_id in existing_meta:
                skipped_cached += 1
                continue

            rgb_png  = rgb_out_dir / f"{tile_id}.png"
            mb_npy   = mb_out_dir  / f"{tile_id}.npy"

            # --- compute window (clamp to image bounds) --------------------
            col_off  = ci * stride
            row_off  = ri * stride
            win_w    = min(tile_size, width  - col_off)
            win_h    = min(tile_size, height - row_off)

            window   = Window(col_off, row_off, win_w, win_h)

            try:
                # Read bands
                rgb_data = rgb_src.read(window=window)            # (3, H, W)
                nir_data = nir_src.read(1, window=window)         # (H, W) float32
                r_data   = r_src.read(1,   window=window)         # (H, W) float32

                # Check valid pixels
                frac = _valid_fraction(rgb_data, nodata_val)
                if frac < min_valid:
                    skipped_nodata += 1
                    continue

                # Pad to full tile_size if tile is on the border
                pad_h = tile_size - win_h
                pad_w = tile_size - win_w
                if pad_h > 0 or pad_w > 0:
                    rgb_data = np.pad(rgb_data, ((0, 0), (0, pad_h), (0, pad_w)))
                    nir_data = np.pad(nir_data, ((0, pad_h), (0, pad_w)))
                    r_data   = np.pad(r_data,   ((0, pad_h), (0, pad_w)))

                # --- RGB PNG (uint8) ----------------------------------------
                rgb_u8 = _normalise_rgb_to_uint8(rgb_data)   # (3, H, W) uint8
                Image.fromarray(rgb_u8.transpose(1, 2, 0)).save(str(rgb_png))

                # --- NDVI ---------------------------------------------------
                ndvi = (nir_data - r_data) / (nir_data + r_data + 1e-8)
                ndvi = ndvi.astype(np.float32)

                # --- Normalise RGB to [0,1] for multiband stack ------------
                rgb_f32 = rgb_u8.astype(np.float32) / 255.0   # (3, H, W)

                # Stack: [R, G, B, NIR, NDVI]  (5, H, W)
                multiband = np.stack([
                    rgb_f32[0], rgb_f32[1], rgb_f32[2],
                    nir_data, ndvi,
                ], axis=0).astype(np.float32)

                np.save(str(mb_npy), multiband)

                # --- Tile geo-transform ------------------------------------
                tile_transform = rasterio.transform.from_origin(
                    transform.c + col_off * transform.a,   # west
                    transform.f + row_off * transform.e,   # north  (e < 0)
                    abs(transform.a),                       # pixel width
                    abs(transform.e),                       # pixel height
                )

                tiles_meta.append({
                    "tile_id":     tile_id,
                    "row_idx":     ri,
                    "col_idx":     ci,
                    "row_off":     row_off,
                    "col_off":     col_off,
                    "actual_h":    win_h,
                    "actual_w":    win_w,
                    "tile_size":   tile_size,
                    "transform":   list(tile_transform)[:6],   # (a,b,c,d,e,f)
                })
                written += 1

            except Exception as exc:
                log.error("Tile %s failed: %s", tile_id, exc, exc_info=True)
                continue

        # Save metadata
        with open(metadata_path, "w", encoding="utf-8") as fh:
            json.dump({"tiles": tiles_meta}, fh, indent=2)

    log.info(
        "\nTiling complete: written=%d  skipped_nodata=%d  cached=%d",
        written, skipped_nodata, skipped_cached,
    )
    log.info("Metadata → %s  (%d tiles total)", metadata_path, len(tiles_meta))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    config_path = Path(__file__).parent / "config.yaml"
    cfg = _load_config(config_path)
    output_dir = Path(cfg["paths"]["output_dir"])
    log = _setup_logger(output_dir / "logs")

    log.info("=" * 60)
    log.info("02_tiling.py — ortomosaico tiling")
    log.info("=" * 60)

    run_tiling(cfg, log)
    log.info("02_tiling.py complete.")


if __name__ == "__main__":
    main()
