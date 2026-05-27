"""
Run the full SAM tree-detection pipeline in order.

Steps
-----
  01_align.py        Align rasters to RGB reference
  02_tiling.py       Tile ortomosaico into 1024×1024 patches
  03_sam_inference.py Run SAM on every RGB tile
  04_filter.py       Filter masks by NDVI, shape, NMS
  05_crop.py         Crop individual tree patches
  06_export.py       Export GeoJSON with geo-coordinates

Usage
-----
  python run_all.py [--start STEP] [--end STEP]

  --start N   begin at step N (1–6, default 1)
  --end   N   stop  at step N (1–6, default 6)
"""

from __future__ import annotations

import argparse
import importlib
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml


STEPS: list[tuple[str, str]] = [
    ("01", "01_align"),
    ("02", "02_tiling"),
    ("03", "03_sam_inference"),
    ("04", "04_filter"),
    ("05", "05_crop"),
    ("06", "06_export"),
]


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def _setup_root_logger(log_dir: Path) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"run_all_{datetime.now():%Y%m%d_%H%M%S}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    return logging.getLogger("run_all")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def _run_step(module_name: str, log: logging.Logger) -> float:
    """Import the step module and call main(). Returns elapsed seconds."""
    log.info("─" * 60)
    log.info(">>> %s", module_name)
    log.info("─" * 60)
    t0 = time.perf_counter()

    # Add pipeline dir to path so imports work
    pipeline_dir = str(Path(__file__).parent)
    if pipeline_dir not in sys.path:
        sys.path.insert(0, pipeline_dir)

    mod = importlib.import_module(module_name)
    # Force reload to avoid cached state between runs
    importlib.reload(mod)
    mod.main()

    elapsed = time.perf_counter() - t0
    log.info("<<< %s finished in %.1f s (%.1f min)", module_name, elapsed, elapsed / 60)
    return elapsed


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full SAM tree pipeline.")
    parser.add_argument("--start", type=int, default=1, metavar="N",
                        help="Start at step N (1–6, default 1)")
    parser.add_argument("--end", type=int, default=6, metavar="N",
                        help="Stop at step N (1–6, default 6)")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    # Load config to find output dir for logs
    config_path = Path(__file__).parent / "config.yaml"
    if not config_path.exists():
        print(f"ERROR: config.yaml not found at {config_path}", file=sys.stderr)
        sys.exit(1)
    with open(config_path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    output_dir = Path(cfg["paths"]["output_dir"])

    log = _setup_root_logger(output_dir / "logs")

    log.info("=" * 60)
    log.info("SAM Tree Detection Pipeline")
    log.info("Steps %d → %d", args.start, args.end)
    log.info("=" * 60)

    selected = [
        (num, name) for num, name in STEPS
        if args.start <= int(num) <= args.end
    ]

    if not selected:
        log.error("No steps selected (--start %d --end %d). Exiting.", args.start, args.end)
        sys.exit(1)

    timings: list[tuple[str, float]] = []
    t_total = time.perf_counter()

    for step_num, module_name in selected:
        try:
            elapsed = _run_step(module_name, log)
            timings.append((module_name, elapsed))
        except Exception as exc:
            log.error("Step %s failed: %s", module_name, exc, exc_info=True)
            log.error("Pipeline aborted at step %s.", step_num)
            sys.exit(1)

    total_elapsed = time.perf_counter() - t_total

    log.info("\n" + "=" * 60)
    log.info("Pipeline complete — summary")
    log.info("=" * 60)
    for name, t in timings:
        log.info("  %-30s  %6.1f s  (%5.1f min)", name, t, t / 60)
    log.info("  %s", "─" * 42)
    log.info("  %-30s  %6.1f s  (%5.1f min)", "TOTAL", total_elapsed, total_elapsed / 60)
    log.info("=" * 60)


if __name__ == "__main__":
    main()
