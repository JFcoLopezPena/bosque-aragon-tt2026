"""
robustness_test.py
Prueba de robustez espectral del modelo de clasificacion fine-tuneado.

Aplica 7 perturbaciones espectrales al test set y compara metricas
contra la linea base para evaluar generalizacion.

Nota: los crops son RGB; la simulacion NDVI opera sobre el canal verde
(reflectancia de vegetacion en RGB, proxy del NIR en imagen multiespectral).

Uso:
    python robustness_test.py
    python robustness_test.py --config config.yaml
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Callable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import autocast
from tqdm import tqdm
import yaml

sys.path.insert(0, str(Path(__file__).parent))
from dataset import create_dataloaders
from model   import ClassificationSAM

# ── Rutas ─────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent.parent
SAM_CKPT    = BASE_DIR / "checkpoints" / "sam_vit_b_01ec64.pth"
CLS_CKPT    = BASE_DIR / "checkpoints" / "cls_finetuned" / "best_model.pth"
RESULTS_DIR = BASE_DIR / "results" / "clasificacion"

THRESHOLD_MASK = 0.65


# ── Perturbaciones ─────────────────────────────────────────────────────────────
# Todas operan sobre tensores [B, 3, H, W] en [0, 1], devuelven mismo shape.

def perturb_ndvi_low(images: torch.Tensor) -> torch.Tensor:
    """Reduccion espectral leve: canal verde (proxy NIR en RGB) x0.90."""
    out = images.clone()
    out[:, 1, :, :] = (out[:, 1, :, :] * 0.90).clamp(0.0, 1.0)
    return out


def perturb_ndvi_severe(images: torch.Tensor) -> torch.Tensor:
    """Reduccion espectral severa: canal verde x0.75."""
    out = images.clone()
    out[:, 1, :, :] = (out[:, 1, :, :] * 0.75).clamp(0.0, 1.0)
    return out


def perturb_gaussian_noise(images: torch.Tensor, sigma: float = 0.05) -> torch.Tensor:
    """Ruido gaussiano (mean=0, sigma=0.05) en todos los canales."""
    noise = torch.randn_like(images) * sigma
    return (images + noise).clamp(0.0, 1.0)


def perturb_brightness_up(images: torch.Tensor, factor: float = 1.15) -> torch.Tensor:
    """Sobreexposicion: brillo x1.15."""
    return (images * factor).clamp(0.0, 1.0)


def perturb_brightness_down(images: torch.Tensor, factor: float = 0.85) -> torch.Tensor:
    """Subexposicion: brillo x0.85."""
    return (images * factor).clamp(0.0, 1.0)


def perturb_combined(images: torch.Tensor) -> torch.Tensor:
    """NDVI_LOW + ruido gaussiano simultaneamente."""
    return perturb_gaussian_noise(perturb_ndvi_low(images))


# Cada entrada: (label, descripcion, funcion, bloque)
# bloque "real"        → condiciones plausibles en segunda toma con DJI Mavic 3M
# bloque "adversarial" → condiciones artificiales / fuera del escenario de vuelo
PERTURBATIONS: list[tuple[str, str, Callable, str]] = [
    ("Original",       "Linea base (sin perturbacion)",           lambda x: x,              "baseline"),
    ("NDVI -10%",      "Canal verde x0.90 — estres hidrico leve", perturb_ndvi_low,          "real"),
    ("NDVI -25%",      "Canal verde x0.75 — estres avanzado",     perturb_ndvi_severe,       "real"),
    ("Brillo +15%",    "Sobrerexposicion (hora distinta, sol)",    perturb_brightness_up,     "real"),
    ("Brillo -15%",    "Subexposicion (nublado, sombras)",         perturb_brightness_down,   "real"),
    ("Ruido gaussiano","N(0, 0.05) en todos los canales",          perturb_gaussian_noise,    "adversarial"),
    ("Combinada",      "NDVI -10% + ruido gaussiano",              perturb_combined,          "adversarial"),
]


# ── Metricas ──────────────────────────────────────────────────────────────────

def compute_metrics(preds: torch.Tensor, targets: torch.Tensor) -> dict:
    acc   = (preds == targets).float().mean().item()
    total = len(targets)
    out   = {"accuracy": acc, "n_total": total}
    f1s   = {}
    for cls_idx, cls_name in ((0, "sano"), (1, "estres")):
        tp = ((preds == cls_idx) & (targets == cls_idx)).sum().float()
        fp = ((preds == cls_idx) & (targets != cls_idx)).sum().float()
        fn = ((preds != cls_idx) & (targets == cls_idx)).sum().float()
        prec = (tp / (tp + fp + 1e-8)).item()
        rec  = (tp / (tp + fn + 1e-8)).item()
        f1   = 2 * prec * rec / (prec + rec + 1e-8)
        out[f"precision_{cls_name}"] = prec
        out[f"recall_{cls_name}"]    = rec
        out[f"f1_{cls_name}"]        = f1
        f1s[cls_idx] = f1
    out["f1_macro"] = (f1s[0] + f1s[1]) / 2
    return out


# ── Inferencia con perturbacion ───────────────────────────────────────────────

@torch.no_grad()
def run_perturbed(
    model:      ClassificationSAM,
    loader:     torch.utils.data.DataLoader,
    perturb_fn: Callable,
    device:     torch.device,
    use_amp:    bool,
    image_size: int,
    label:      str,
) -> dict:
    all_preds   = []
    all_targets = []

    for images, masks, labels in tqdm(loader, desc=f"  {label:<22}", leave=False,
                                       unit="batch", ncols=80):
        images = perturb_fn(images).to(device)
        labels = labels.to(device)

        images_1024 = F.interpolate(images, (1024, 1024),
                                    mode="bilinear", align_corners=False)

        with autocast("cuda", enabled=(use_amp and device.type == "cuda")):
            logits_cls, _ = model(images_1024, output_size=(image_size, image_size))

        preds = logits_cls.argmax(dim=1)
        all_preds.append(preds.cpu())
        all_targets.append(labels.cpu())

    return compute_metrics(torch.cat(all_preds), torch.cat(all_targets))


# ── Tabla de resultados ───────────────────────────────────────────────────────

def _print_block(title: str, rows: list[dict], baseline_f1: float, threshold: float | None) -> None:
    hdr = (f"  {'Condicion':<22} | {'Accuracy':>8} | {'F1 Macro':>8} | "
           f"{'F1 Estres':>9} | {'Prec.':>6} | {'Recall':>6} | {'Delta F1':>9}")
    sep = "  " + "-" * (len(hdr) - 2)
    thr_note = f"  (umbral robustez: Delta F1 >= -{threshold:.2f})" if threshold else "  (referencia informativa)"
    print(f"\n  {'='*70}")
    print(f"  {title}")
    print(f"{thr_note}")
    print(sep)
    print(hdr)
    print(sep)
    for r in rows:
        delta = r["f1_macro"] - baseline_f1
        sign  = "+" if delta >= 0 else ""
        print(
            f"  {r['label']:<22} | {r['accuracy']:>8.4f} | {r['f1_macro']:>8.4f} | "
            f"{r['f1_estres']:>9.4f} | {r['precision_estres']:>6.4f} | "
            f"{r['recall_estres']:>6.4f} | {sign}{delta:>8.4f}"
        )
    print(sep + "\n")


def print_table(rows: list[dict], baseline_f1: float) -> None:
    baseline_row = [r for r in rows if r["block"] == "baseline"]
    real_rows    = [r for r in rows if r["block"] == "real"]
    adv_rows     = [r for r in rows if r["block"] == "adversarial"]

    _print_block("BLOQUE 1 — Condiciones de vuelo reales (DJI Mavic 3M)",
                 baseline_row + real_rows, baseline_f1, threshold=0.10)
    _print_block("BLOQUE 2 — Condiciones adversariales (artificiales)",
                 baseline_row + adv_rows, baseline_f1, threshold=None)


# ── Grafica ───────────────────────────────────────────────────────────────────

def _bar_color(block: str, delta: float) -> str:
    if block == "baseline":
        return "#2c7bb6"
    if block == "real":
        return "#1a9641" if delta >= -0.10 else "#d7191c"
    # adversarial: escala de gris azulado para diferenciar visualmente
    return "#7b9ec4" if delta >= -0.10 else "#a05070"


def plot_results(rows: list[dict], baseline_f1: float, out_path: Path) -> None:
    from matplotlib.patches import Patch, FancyBboxPatch

    real_rows = [r for r in rows if r["block"] in ("baseline", "real")]
    adv_rows  = [r for r in rows if r["block"] in ("baseline", "adversarial")]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5),
                             gridspec_kw={"width_ratios": [5, 3]})
    fig.suptitle("Prueba de Robustez Espectral — SAM Fine-tuneado R2 | TT 2026-A127 ESCOM-IPN",
                 fontsize=12, fontweight="bold", y=1.01)

    def draw_block(ax, block_rows, title, threshold_line: bool):
        labels = [r["label"] for r in block_rows]
        f1s    = [r["f1_macro"] for r in block_rows]
        colors = [_bar_color(r["block"], r["f1_macro"] - baseline_f1) for r in block_rows]

        bars = ax.bar(range(len(labels)), f1s, color=colors, width=0.55, zorder=3)
        ax.axhline(baseline_f1, color="#2c7bb6", linestyle="--", linewidth=1.5,
                   zorder=4, label=f"Baseline F1={baseline_f1:.3f}")

        if threshold_line:
            ax.axhline(baseline_f1 - 0.10, color="#d7191c", linestyle=":",
                       linewidth=1.2, zorder=4, label="Umbral robustez (-10 pp)")

        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=18, ha="right", fontsize=9.5)
        ax.set_ylabel("F1 Macro", fontsize=10)
        all_f1 = [baseline_f1] + f1s
        ax.set_ylim(max(0.0, min(all_f1) - 0.14), min(1.0, max(all_f1) + 0.09))
        ax.set_title(title, fontsize=10.5, fontweight="bold", pad=8)
        ax.yaxis.grid(True, linestyle=":", alpha=0.55, zorder=0)
        ax.set_axisbelow(True)
        ax.legend(fontsize=8.5, loc="lower right")

        for bar, val, row in zip(bars, f1s, block_rows):
            delta = val - baseline_f1
            sign  = "+" if delta >= 0 else ""
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
            if row["block"] != "baseline":
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() / 2,
                        f"{sign}{delta:.3f}", ha="center", va="center",
                        fontsize=7.5, color="white", fontweight="bold")

    draw_block(axes[0], real_rows,
               "Bloque 1 — Condiciones de vuelo reales\n(DJI Mavic 3 Multispectral)",
               threshold_line=True)
    draw_block(axes[1], adv_rows,
               "Bloque 2 — Condiciones adversariales\n(referencia informativa)",
               threshold_line=False)

    # Leyenda de colores compartida
    from matplotlib.patches import Patch as P
    legend_handles = [
        P(color="#1a9641", label="Real — Delta >= -0.10 (robusto)"),
        P(color="#d7191c", label="Real — Delta < -0.10 (sensible)"),
        P(color="#7b9ec4", label="Adversarial — Delta >= -0.10"),
        P(color="#a05070", label="Adversarial — Delta < -0.10"),
    ]
    fig.legend(handles=legend_handles, fontsize=8.5, loc="lower center",
               ncol=4, bbox_to_anchor=(0.5, -0.06))

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Robust] Grafica guardada: {out_path}")


# ── Veredictos ────────────────────────────────────────────────────────────────

def _verdict_real(rows: list[dict], baseline_f1: float) -> str:
    real_rows = [r for r in rows if r["block"] == "real"]
    deltas    = [r["f1_macro"] - baseline_f1 for r in real_rows]
    worst     = min(deltas) if deltas else 0.0

    print("=" * 72)
    print("  VEREDICTO 1 — CONDICIONES DE VUELO REALES")
    print("  (variaciones plausibles con DJI Mavic 3 Multispectral)")
    print("=" * 72)
    print(f"  Peor caida F1 Macro: {worst:+.4f} ({worst*100:+.1f} pp)  "
          f"[umbral: -0.10]\n")

    if worst >= -0.10:
        verdict = "ROBUSTO"
        marker  = "OK"
        msg = ("El modelo mantiene rendimiento ante variaciones espectrales reales.\n"
               "  La degradacion maxima en condiciones de vuelo es < 10 pp en F1 Macro.\n"
               "  Conclusion: apto para uso en segunda toma o condiciones climaticas distintas.")
    else:
        verdict = "SENSIBLE A CONDICIONES REALES"
        marker  = "FAIL"
        worst_label = real_rows[deltas.index(worst)]["label"]
        msg = (f"Caida > 10 pp en '{worst_label}' bajo condiciones de vuelo plausibles.\n"
               "  Recomendacion: incluir augmentaciones de brillo/NDVI en entrenamiento.")

    print(f"  [{marker}] {verdict}")
    print(f"  {msg}")
    print("=" * 72 + "\n")
    return verdict


def _verdict_adversarial(rows: list[dict], baseline_f1: float) -> str:
    adv_rows = [r for r in rows if r["block"] == "adversarial"]
    deltas   = [r["f1_macro"] - baseline_f1 for r in adv_rows]
    worst    = min(deltas) if deltas else 0.0

    print("=" * 72)
    print("  VEREDICTO 2 — CONDICIONES ADVERSARIALES (informativo)")
    print("  (NO representan condiciones normales de vuelo — referencia)")
    print("=" * 72)
    print(f"  Peor caida F1 Macro: {worst:+.4f} ({worst*100:+.1f} pp)\n")

    if worst >= -0.10:
        verdict = "ESTABLE INCLUSO BAJO CONDICIONES ARTIFICIALES"
        marker  = "INFO"
        msg = "Robustez notable: el modelo no se degrada significativamente ante ruido extremo."
    elif worst >= -0.20:
        verdict = "DEGRADACION MODERADA BAJO CONDICIONES ARTIFICIALES"
        marker  = "INFO"
        msg = ("Caida entre 10 y 20 pp bajo perturbaciones fuera del escenario real.\n"
               "  Comportamiento esperado — no compromete la validez del modelo.")
    else:
        verdict = "DEGRADACION ALTA BAJO CONDICIONES ARTIFICIALES"
        marker  = "INFO"
        msg = ("Caida > 20 pp bajo condiciones artificiales extremas.\n"
               "  Esperado para perturbaciones fuera del dominio de entrenamiento.\n"
               "  No afecta la evaluacion en condiciones de vuelo reales.")

    print(f"  [{marker}] {verdict}")
    print(f"  {msg}")
    print("=" * 72 + "\n")
    return verdict


def print_verdict(rows: list[dict], baseline_f1: float) -> tuple[str, str]:
    v_real = _verdict_real(rows, baseline_f1)
    v_adv  = _verdict_adversarial(rows, baseline_f1)
    return v_real, v_adv


# ── Main ──────────────────────────────────────────────────────────────────────

def run(config_path: str = "config.yaml") -> None:
    t_start = time.time()

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp    = cfg["training"]["mixed_precision"]
    image_size = cfg["training"]["image_size"]

    # ── Info inicial ──────────────────────────────────────────────────────────
    print("=" * 72)
    print("  PRUEBA DE ROBUSTEZ ESPECTRAL — SAM Fine-tuneado R2")
    print("  TT 2026-A127, ESCOM-IPN")
    print("=" * 72)
    print(f"  Dispositivo : {device}")
    if device.type == "cuda":
        free, total = torch.cuda.mem_get_info()
        print(f"  VRAM        : {free/1024**3:.1f} GB libre / {total/1024**3:.1f} GB total")
        print(f"  GPU         : {torch.cuda.get_device_name(0)}")
    print(f"  AMP         : {use_amp}")
    print(f"  Perturbaciones: {len(PERTURBATIONS)}")
    print(f"  Tiempo estimado: ~5-8 min segun GPU\n")

    # ── Cargar modelo ─────────────────────────────────────────────────────────
    print(f"[Robust] Cargando modelo desde {CLS_CKPT} ...")
    if not CLS_CKPT.exists():
        raise FileNotFoundError(f"Checkpoint no encontrado: {CLS_CKPT}")

    model = ClassificationSAM.load_checkpoint(
        ft_checkpoint=CLS_CKPT,
        sam_checkpoint=SAM_CKPT,
        unfreeze_encoder_blocks=cfg["model"].get("unfreeze_encoder_blocks", 2),
        num_classes=cfg["model"]["num_classes"],
    ).to(device)
    model.eval()

    ckpt = torch.load(str(CLS_CKPT), map_location="cpu", weights_only=False)
    meta = ckpt.get("metadata", {})
    if meta:
        print(f"[Robust] Checkpoint: epoch={meta.get('epoch','?')} "
              f"val_F1={meta.get('f1_enfermo', '?')}")

    # ── Cargar test set ───────────────────────────────────────────────────────
    print("[Robust] Cargando test set ...")
    _, _, test_loader = create_dataloaders(cfg)
    n_test = len(test_loader.dataset)
    print(f"[Robust] Test set: {n_test} imagenes\n")

    # ── Ejecutar perturbaciones ───────────────────────────────────────────────
    results_raw: list[dict] = []

    for label, description, fn, block in PERTURBATIONS:
        print(f"[{label}] {description}")
        t0 = time.time()
        m  = run_perturbed(model, test_loader, fn, device, use_amp, image_size, label)
        elapsed = time.time() - t0
        results_raw.append({
            "label":            label,
            "description":      description,
            "block":            block,
            "accuracy":         m["accuracy"],
            "f1_macro":         m["f1_macro"],
            "f1_estres":        m["f1_estres"],
            "precision_estres": m["precision_estres"],
            "recall_estres":    m["recall_estres"],
            "f1_sano":          m["f1_sano"],
            "elapsed_s":        round(elapsed, 1),
        })
        print(f"       Accuracy={m['accuracy']:.4f}  F1_macro={m['f1_macro']:.4f}  "
              f"F1_estres={m['f1_estres']:.4f}  ({elapsed:.1f}s)")

    # ── Tabla comparativa (dos bloques) ───────────────────────────────────────
    baseline_f1 = results_raw[0]["f1_macro"]
    print_table(results_raw, baseline_f1)

    # ── Veredictos ────────────────────────────────────────────────────────────
    verdict_real, verdict_adv = print_verdict(results_raw, baseline_f1)

    # ── Grafica ───────────────────────────────────────────────────────────────
    png_path = RESULTS_DIR / "robustness_test.png"
    plot_results(results_raw, baseline_f1, png_path)

    # ── Exportar JSON ─────────────────────────────────────────────────────────
    json_path = RESULTS_DIR / "robustness_results.json"
    export = {
        "model_checkpoint":    str(CLS_CKPT),
        "n_test_images":       n_test,
        "baseline_f1_macro":   baseline_f1,
        "verdict_real":        verdict_real,
        "verdict_adversarial": verdict_adv,
        "perturbations":       results_raw,
        "total_elapsed_s":     round(time.time() - t_start, 1),
    }
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(export, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[Robust] Resultados JSON: {json_path}")

    total = time.time() - t_start
    print(f"[Robust] Completado en {total/60:.1f} min\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prueba de robustez espectral")
    parser.add_argument("--config", default="config.yaml",
                        help="Ruta al config.yaml (default: config.yaml)")
    args = parser.parse_args()
    os.chdir(Path(__file__).parent)
    run(args.config)
