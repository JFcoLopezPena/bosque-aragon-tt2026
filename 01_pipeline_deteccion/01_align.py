"""
Align NIR, R and MDE rasters to the RGB reference (same CRS, extent, resolution).
Outputs float32 GeoTIFFs in output/aligned/.

NIR and R (16-bit, range 2221–65535) are normalised to float32 [0, 1].
RGB is kept as-is (uint8 or uint16 depending on source).
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import rasterio
import yaml
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform, reproject


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def _setup_logger(log_dir: Path) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    from datetime import datetime
    log_file = log_dir / f"01_align_{datetime.now():%Y%m%d_%H%M%S}.log"
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


def _align_raster(
    src_path: Path,
    dst_path: Path,
    ref_crs: rasterio.crs.CRS,
    ref_transform,
    ref_width: int,
    ref_height: int,
    resampling: Resampling = Resampling.bilinear,
    normalise_to_float: bool = False,
    log: logging.Logger | None = None,
) -> None:
    """Reproject *src_path* to match the reference grid and write to *dst_path*.

    Parameters
    ----------
    normalise_to_float:
        When True, linearly normalise each band to [0, 1] float32 using
        per-band min/max from a representative block scan.
    """
    log = log or logging.getLogger(__name__)
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(src_path) as src:
        log.info(
            "  src: %s  bands=%d  dtype=%s  crs=%s  res=(%.4f,%.4f)",
            src_path.name, src.count, src.dtypes[0], src.crs,
            src.res[0], src.res[1],
        )

        dst_dtype = "float32" if normalise_to_float else src.dtypes[0]

        dst_meta = {
            "driver": "GTiff",
            "height": ref_height,
            "width": ref_width,
            "count": src.count,
            "dtype": dst_dtype,
            "crs": ref_crs,
            "transform": ref_transform,
            "compress": "lzw",
            "tiled": True,
            "blockxsize": 512,
            "blockysize": 512,
        }

        with rasterio.open(dst_path, "w", **dst_meta) as dst:
            for band_idx in range(1, src.count + 1):
                # --- reproject band into a temporary float32 array ----------
                reprojected = np.zeros((ref_height, ref_width), dtype="float32")
                reproject(
                    source=rasterio.band(src, band_idx),
                    destination=reprojected,
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=ref_transform,
                    dst_crs=ref_crs,
                    resampling=resampling,
                )

                if normalise_to_float:
                    valid = reprojected[reprojected != 0]
                    if valid.size > 0:
                        vmin, vmax = valid.min(), valid.max()
                        if vmax > vmin:
                            reprojected = (reprojected - vmin) / (vmax - vmin)
                            reprojected = np.clip(reprojected, 0.0, 1.0)
                    dst.write(reprojected.astype("float32"), band_idx)
                else:
                    dst.write(reprojected.astype(dst_dtype), band_idx)

    log.info(
        "  dst: %s  size=(%d×%d)  dtype=%s",
        dst_path.name, ref_width, ref_height, dst_dtype,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    config_path = Path(__file__).parent / "config.yaml"
    cfg = _load_config(config_path)

    output_dir = Path(cfg["paths"]["output_dir"])
    log = _setup_logger(output_dir / "logs")

    log.info("=" * 60)
    log.info("01_align.py — raster alignment")
    log.info("=" * 60)

    # Input paths
    rgb_path = Path(cfg["paths"]["rgb_tif"])
    nir_path = Path(cfg["paths"]["nir_tif"])
    r_path   = Path(cfg["paths"]["r_tif"])
    mde_path = Path(cfg["paths"]["mde_tif"])

    for p, label in [(rgb_path, "RGB"), (nir_path, "NIR"), (r_path, "R"), (mde_path, "MDE")]:
        _assert_exists(p, label)

    aligned_dir = output_dir / "aligned"
    aligned_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Read RGB reference metadata
    # ------------------------------------------------------------------
    log.info("\n[reference] Opening RGB raster …")
    with rasterio.open(rgb_path) as rgb:
        ref_crs       = rgb.crs
        ref_transform = rgb.transform
        ref_width     = rgb.width
        ref_height    = rgb.height
        ref_res       = rgb.res

        log.info(
            "  RGB: bands=%d  dtype=%s  size=(%d×%d)  res=(%.4f,%.4f)  crs=%s",
            rgb.count, rgb.dtypes[0], ref_width, ref_height,
            ref_res[0], ref_res[1], ref_crs,
        )

        if ref_crs.to_epsg() != 32614:
            log.warning("  RGB CRS is %s, expected EPSG:32614!", ref_crs)

    # ------------------------------------------------------------------
    # Copy / link RGB (already the reference — just copy to aligned/)
    # ------------------------------------------------------------------
    rgb_out = aligned_dir / "rgb_aligned.tif"
    if rgb_out.exists():
        log.info("\n[RGB] %s already exists, skipping.", rgb_out.name)
    else:
        log.info("\n[RGB] Copying to aligned directory …")
        import shutil
        shutil.copy2(rgb_path, rgb_out)
        log.info("  Copied %s → %s", rgb_path.name, rgb_out.name)

    # ------------------------------------------------------------------
    # Align NIR  (normalise to float32)
    # ------------------------------------------------------------------
    nir_out = aligned_dir / "nir_aligned.tif"
    if nir_out.exists():
        log.info("\n[NIR] %s already exists, skipping.", nir_out.name)
    else:
        log.info("\n[NIR] Reprojecting and normalising …")
        _align_raster(
            nir_path, nir_out,
            ref_crs, ref_transform, ref_width, ref_height,
            normalise_to_float=True, log=log,
        )

    # ------------------------------------------------------------------
    # Align R band  (normalise to float32)
    # ------------------------------------------------------------------
    r_out = aligned_dir / "r_aligned.tif"
    if r_out.exists():
        log.info("\n[R] %s already exists, skipping.", r_out.name)
    else:
        log.info("\n[R] Reprojecting and normalising …")
        _align_raster(
            r_path, r_out,
            ref_crs, ref_transform, ref_width, ref_height,
            normalise_to_float=True, log=log,
        )

    # ------------------------------------------------------------------
    # Align MDE (keep native dtype, no normalisation)
    # ------------------------------------------------------------------
    mde_out = aligned_dir / "mde_aligned.tif"
    if mde_out.exists():
        log.info("\n[MDE] %s already exists, skipping.", mde_out.name)
    else:
        log.info("\n[MDE] Reprojecting …")
        _align_raster(
            mde_path, mde_out,
            ref_crs, ref_transform, ref_width, ref_height,
            normalise_to_float=False, log=log,
        )

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------
    log.info("\n--- Aligned rasters ---")
    for out_path in [rgb_out, nir_out, r_out, mde_out]:
        with rasterio.open(out_path) as ds:
            log.info(
                "  %-25s  bands=%-2d  dtype=%-8s  size=(%d×%d)",
                out_path.name, ds.count, ds.dtypes[0], ds.width, ds.height,
            )

    log.info("\n01_align.py complete.")


if __name__ == "__main__":
    main()
