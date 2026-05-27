"""
Intervalos de confianza mediante bootstrapping para las metricas del modelo
de clasificacion fitosanitaria fine-tuneado (R2).

Flujo:
  1. Cargar modelo + test set desde config.yaml
  2. Una sola pasada de inferencia -> pred_labels, true_labels
  3. 1000 remuestreos con reemplazo sobre los pares (pred, true)
  4. IC 95% y 99% por percentiles
  5. Figura + JSON + texto para tesis

Uso:
    python bootstrap_confidence.py
    python bootstrap_confidence.py --config path/to/config.yaml
    python bootstrap_confidence.py --n_bootstrap 2000
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import autocast
import yaml

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, str(Path(__file__).parent))
from dataset import create_dataloaders
from model   import ClassificationSAM

# ---------------------------------------------------------------------------
# Metricas sobre arrays numpy (rapidas para bootstrapping)
# ---------------------------------------------------------------------------

def _safe_div(a: float, b: float) -> float:
    return a / b if b > 1e-9 else 0.0


def compute_metrics_np(preds: np.ndarray, targets: np.ndarray) -> dict[str, float]:
    n   = len(targets)
    acc = float(np.mean(preds == targets))

    results: dict[str, float] = {"accuracy": acc}
    f1s = {}

    for cls_idx, cls_name in ((0, "sano"), (1, "estres")):
        tp = float(np.sum((preds == cls_idx) & (targets == cls_idx)))
        fp = float(np.sum((preds == cls_idx) & (targets != cls_idx)))
        fn = float(np.sum((preds != cls_idx) & (targets == cls_idx)))
        tn = float(np.sum((preds != cls_idx) & (targets != cls_idx)))

        prec = _safe_div(tp, tp + fp)
        rec  = _safe_div(tp, tp + fn)
        f1   = _safe_div(2 * prec * rec, prec + rec)

        results[f"precision_{cls_name}"] = prec
        results[f"recall_{cls_name}"]    = rec
        results[f"f1_{cls_name}"]        = f1
        f1s[cls_idx] = f1

    results["f1_macro"] = (f1s[0] + f1s[1]) / 2
    return results


# ---------------------------------------------------------------------------
# Bootstrapping
# ---------------------------------------------------------------------------

def run_bootstrap(
    preds:       np.ndarray,
    targets:     np.ndarray,
    n_bootstrap: int = 1000,
    seed:        int = 42,
) -> dict[str, list[float]]:
    rng = np.random.default_rng(seed)
    n   = len(preds)
    metric_keys = [
        "accuracy", "f1_macro", "f1_estres",
        "precision_estres", "recall_estres", "f1_sano",
    ]
    samples: dict[str, list[float]] = {k: [] for k in metric_keys}

    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        m   = compute_metrics_np(preds[idx], targets[idx])
        for k in metric_keys:
            samples[k].append(m[k])

    return samples


def ci(values: list[float], level: float = 95) -> tuple[float, float]:
    alpha = (100 - level) / 2
    return (float(np.percentile(values, alpha)),
            float(np.percentile(values, 100 - alpha)))


# ---------------------------------------------------------------------------
# Figura
# ---------------------------------------------------------------------------

METRIC_LABELS = {
    "accuracy"        : "Accuracy",
    "f1_macro"        : "F1 macro",
    "f1_estres"       : "F1 estres severo",
    "precision_estres": "Precision estres",
    "recall_estres"   : "Recall estres",
}

PANEL_METRICS = ["f1_macro", "f1_estres", "recall_estres", "precision_estres", "accuracy"]
PANEL_COLORS  = ["#2980b9", "#c0392b", "#27ae60", "#8e44ad", "#e67e22"]


def plot_bootstrap(
    samples:       dict[str, list[float]],
    point_metrics: dict[str, float],
    out_path:      Path,
) -> None:
    n_panels = len(PANEL_METRICS)
    fig, axes = plt.subplots(1, n_panels, figsize=(4 * n_panels, 4.5))
    fig.suptitle(
        "Distribuciones Bootstrap (n=1,000 remuestreos)",
        fontsize=13, fontweight="bold", y=1.02,
    )

    for ax, metric, color in zip(axes, PANEL_METRICS, PANEL_COLORS):
        vals  = np.array(samples[metric])
        point = point_metrics[metric]
        lo95, hi95 = ci(list(vals), 95)

        ax.hist(vals, bins=40, color=color, alpha=0.55, edgecolor="none")

        # IC 95% shaded region via axvspan on x-axis
        ax.axvspan(lo95, hi95, alpha=0.20, color=color, label="IC 95%")
        ax.axvline(point,  color="black",  linewidth=2.0, linestyle="-",  label=f"Puntual: {point:.4f}")
        ax.axvline(lo95,   color=color,    linewidth=1.2, linestyle="--")
        ax.axvline(hi95,   color=color,    linewidth=1.2, linestyle="--")

        ax.set_title(METRIC_LABELS.get(metric, metric), fontsize=10, fontweight="bold")
        ax.set_xlabel("Valor metrica", fontsize=9)
        ax.set_ylabel("Frecuencia", fontsize=9)
        ax.legend(fontsize=7.5, loc="upper left")

        # Annotation with CI values
        ax.text(
            0.97, 0.97,
            f"IC95: [{lo95:.3f}, {hi95:.3f}]",
            transform=ax.transAxes,
            ha="right", va="top", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        )

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figura guardada: {out_path}")


# ---------------------------------------------------------------------------
# Tabla de resultados
# ---------------------------------------------------------------------------

TABLA_METRICS = [
    ("accuracy",         "Accuracy"),
    ("f1_macro",         "F1 macro"),
    ("f1_estres",        "F1 estres severo"),
    ("precision_estres", "Precision estres"),
    ("recall_estres",    "Recall estres"),
    ("f1_sano",          "F1 sano"),
]


def print_table(point_metrics: dict, samples: dict) -> None:
    header = f"  {'Metrica':<22} | {'Puntual':>8} | {'IC 95%':^22} | {'IC 99%':^22}"
    sep    = "  " + "-" * (len(header) - 2)
    print(sep)
    print(header)
    print(sep)
    for key, label in TABLA_METRICS:
        pv      = point_metrics[key]
        lo95, hi95 = ci(samples[key], 95)
        lo99, hi99 = ci(samples[key], 99)
        ic95   = f"[{lo95:.3f}, {hi95:.3f}]"
        ic99   = f"[{lo99:.3f}, {hi99:.3f}]"
        print(f"  {label:<22} | {pv:>8.4f} | {ic95:^22} | {ic99:^22}")
    print(sep)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(config_path: str = "config.yaml", n_bootstrap: int = 1000) -> None:
    os.chdir(Path(__file__).parent)

    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    OUT_DIR = Path("C:/Users/fcolo/Desktop/TT/results/clasificacion/bootstrap_ci")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = config["training"]["mixed_precision"]
    img_sz  = config["training"]["image_size"]

    # ── 1. Cargar modelo ──────────────────────────────────────────────────────
    ckpt_path = Path(config["paths"]["checkpoint_dir"]) / "best_model.pth"
    if not ckpt_path.exists():
        sys.exit(f"[ERROR] Checkpoint no encontrado: {ckpt_path}")

    print(f"[Bootstrap] Cargando modelo: {ckpt_path}")
    model = ClassificationSAM.load_checkpoint(
        ft_checkpoint=ckpt_path,
        sam_checkpoint=config["model"]["sam_checkpoint"],
        unfreeze_encoder_blocks=config["model"].get("unfreeze_encoder_blocks", 2),
        num_classes=config["model"]["num_classes"],
    ).to(device)
    model.eval()

    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    meta = ckpt.get("metadata", {})
    if meta:
        print(f"[Bootstrap] Checkpoint epoch={meta.get('epoch','?')}, "
              f"F1_enfermo={meta.get('f1_enfermo','?')}")

    # ── 2. Inferencia unica sobre test set ────────────────────────────────────
    print("[Bootstrap] Cargando test set ...")
    _, _, test_loader = create_dataloaders(config)
    n_test = len(test_loader.dataset)
    print(f"[Bootstrap] Test set: {n_test} muestras")

    all_preds  = []
    all_labels = []

    print("[Bootstrap] Inferencia sobre test set (una sola vez) ...")
    with torch.no_grad():
        for batch_idx, (images, masks, labels) in enumerate(test_loader):
            images = images.to(device)
            images_1024 = F.interpolate(images, (1024, 1024), mode="bilinear", align_corners=False)

            with autocast("cuda", enabled=use_amp):
                logits_cls, _ = model(images_1024, output_size=(img_sz, img_sz))

            preds = logits_cls.argmax(dim=1).cpu().numpy()
            all_preds.append(preds)
            all_labels.append(labels.numpy())

            if (batch_idx + 1) % 50 == 0:
                done = (batch_idx + 1) * config["training"]["batch_size"]
                print(f"  {min(done, n_test)}/{n_test} imagenes procesadas")

    preds_np   = np.concatenate(all_preds)
    targets_np = np.concatenate(all_labels)
    print(f"[Bootstrap] Inferencia completada: {len(preds_np)} predicciones")

    # Metricas puntuales (sobre el test set completo)
    # evaluate.py usa "enfermo"; renombramos a "estres" para el bootstrap
    point_raw = compute_metrics_np(preds_np, targets_np)
    point_metrics = {
        "accuracy"        : point_raw["accuracy"],
        "f1_macro"        : point_raw["f1_macro"],
        "f1_estres"       : point_raw["f1_estres"],
        "precision_estres": point_raw["precision_estres"],
        "recall_estres"   : point_raw["recall_estres"],
        "f1_sano"         : point_raw["f1_sano"],
    }

    # ── 3. Bootstrapping ──────────────────────────────────────────────────────
    print(f"[Bootstrap] Remuestreando {n_bootstrap} veces (n={len(preds_np)}) ...")
    samples = run_bootstrap(preds_np, targets_np, n_bootstrap=n_bootstrap)
    print("[Bootstrap] Bootstrapping completado.")

    # ── 4. Tabla de resultados ────────────────────────────────────────────────
    print()
    print("=" * 80)
    print("  INTERVALOS DE CONFIANZA BOOTSTRAP")
    print(f"  Test set: {n_test} ejemplares | Remuestreos: {n_bootstrap}")
    print("=" * 80)
    print_table(point_metrics, samples)
    print()

    # ── 5. Figura ─────────────────────────────────────────────────────────────
    fig_path = OUT_DIR / "bootstrap_distributions.png"
    plot_bootstrap(samples, point_metrics, fig_path)

    # ── 6. JSON ───────────────────────────────────────────────────────────────
    results_out: dict = {
        "n_test"      : int(n_test),
        "n_bootstrap" : n_bootstrap,
        "point_metrics": {k: round(v, 6) for k, v in point_metrics.items()},
        "confidence_intervals": {},
    }
    for key, _ in TABLA_METRICS:
        lo95, hi95 = ci(samples[key], 95)
        lo99, hi99 = ci(samples[key], 99)
        results_out["confidence_intervals"][key] = {
            "mean_bootstrap": round(float(np.mean(samples[key])), 6),
            "std_bootstrap" : round(float(np.std(samples[key])),  6),
            "ic_95"         : [round(lo95, 6), round(hi95, 6)],
            "ic_99"         : [round(lo99, 6), round(hi99, 6)],
        }

    json_path = OUT_DIR / "bootstrap_results.json"
    json_path.write_text(
        json.dumps(results_out, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"  JSON guardado: {json_path}")

    # ── 7. Texto para tesis ───────────────────────────────────────────────────
    lo95_f1e, hi95_f1e = ci(samples["f1_estres"], 95)
    lo95_rec, hi95_rec = ci(samples["recall_estres"], 95)
    lo95_acc, hi95_acc = ci(samples["accuracy"], 95)
    lo95_mac, hi95_mac = ci(samples["f1_macro"], 95)

    print()
    print("=" * 80)
    print("  TEXTO PARA TESIS (copiar directamente)")
    print("=" * 80)
    print()
    print(
        f"Las metricas reportadas corresponden a valores puntuales sobre el conjunto\n"
        f"de prueba de {n_test:,} ejemplares. El intervalo de confianza al 95 %\n"
        f"para el F1-score de la clase estres severo es [{lo95_f1e:.3f}, {hi95_f1e:.3f}],\n"
        f"para el Recall de estres es [{lo95_rec:.3f}, {hi95_rec:.3f}],\n"
        f"para el F1 macro es [{lo95_mac:.3f}, {hi95_mac:.3f}] y\n"
        f"para la exactitud es [{lo95_acc:.3f}, {hi95_acc:.3f}],\n"
        f"estimados mediante bootstrapping con {n_bootstrap:,} remuestreos (n={n_test:,})."
    )
    print()
    print("=" * 80)
    print("[Bootstrap] Listo.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",      default="config.yaml")
    parser.add_argument("--n_bootstrap", type=int, default=1000)
    args = parser.parse_args()
    main(args.config, args.n_bootstrap)
