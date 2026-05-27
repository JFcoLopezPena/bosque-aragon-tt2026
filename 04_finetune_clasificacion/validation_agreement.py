"""
Concordancia entre etiquetas manuales (Label Studio) y clasificacion Qwen2.5-VL.

La exportacion COCO de Label Studio contiene mascaras de segmentacion (no etiquetas
de salud). La validacion mide concordancia en calidad de mascara:
  - Humano: imagen anotada (segmento dibujado) == mascara BUENA
  - Humano: imagen sin anotar                  == mascara cuestionable
  - Qwen:   campo 'mascara' en clasificaciones_r1.json (BUENA/PARCIAL/MALA)

Metrica principal: Cohen's Kappa (binario)
  - Humano-positivo  : tiene anotacion
  - Qwen-positivo    : mascara == BUENA
"""

import json
import re
import sys
from pathlib import Path
from collections import Counter

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ---------------------------------------------------------------------------
# Rutas
# ---------------------------------------------------------------------------
BASE_DIR   = Path(__file__).parent.parent
COCO_FILE  = BASE_DIR / "data" / "segmentacion" / "result.json"
QWEN_FILE  = BASE_DIR / "output" / "clasificaciones_r1.json"
OUT_DIR    = BASE_DIR / "results" / "clasificacion" / "agreement_analysis"

# ---------------------------------------------------------------------------
# Carga de datos
# ---------------------------------------------------------------------------
def load_data():
    if not COCO_FILE.exists():
        sys.exit(f"[ERROR] No encontrado: {COCO_FILE}")
    if not QWEN_FILE.exists():
        sys.exit(f"[ERROR] No encontrado: {QWEN_FILE}")

    coco  = json.loads(COCO_FILE.read_text(encoding="utf-8"))
    qwen  = json.loads(QWEN_FILE.read_text(encoding="utf-8"))
    return coco, qwen


def extract_arbol_id(filename: str) -> str | None:
    m = re.search(r"arbol_\d+_tile_\d+_\d+", filename)
    return m.group(0) if m else None


def build_pairs(coco: dict, qwen: dict) -> list[dict]:
    annotated_image_ids = {a["image_id"] for a in coco["annotations"]}

    pairs = []
    missing = 0
    for img in coco["images"]:
        arbol_id = extract_arbol_id(img["file_name"])
        if arbol_id is None:
            missing += 1
            continue

        qwen_entry = qwen.get(arbol_id)
        if qwen_entry is None:
            missing += 1
            continue

        pairs.append({
            "arbol_id"        : arbol_id,
            "human_annotated" : img["id"] in annotated_image_ids,
            "qwen_mascara"    : qwen_entry.get("mascara", "DESCONOCIDO"),
            "qwen_estado"     : qwen_entry.get("estado",  "DESCONOCIDO"),
        })

    if missing:
        print(f"  [AVISO] {missing} imagenes sin coincidencia omitidas")
    return pairs


# ---------------------------------------------------------------------------
# Metricas
# ---------------------------------------------------------------------------
def cohens_kappa(y_true: list[int], y_pred: list[int]) -> float:
    n = len(y_true)
    if n == 0:
        return float("nan")
    po = sum(a == b for a, b in zip(y_true, y_pred)) / n
    classes = sorted(set(y_true) | set(y_pred))
    pe = 0.0
    for c in classes:
        p_true = y_true.count(c) / n
        p_pred = y_pred.count(c) / n
        pe += p_true * p_pred
    if pe == 1.0:
        return 1.0
    return (po - pe) / (1 - pe)


def binary_metrics(y_true: list[int], y_pred: list[int]) -> dict:
    tp = sum(a == 1 and b == 1 for a, b in zip(y_true, y_pred))
    tn = sum(a == 0 and b == 0 for a, b in zip(y_true, y_pred))
    fp = sum(a == 0 and b == 1 for a, b in zip(y_true, y_pred))
    fn = sum(a == 1 and b == 0 for a, b in zip(y_true, y_pred))
    n  = len(y_true)

    accuracy  = (tp + tn) / n if n else 0
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall    = tp / (tp + fn) if (tp + fn) else 0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
    specificity = tn / (tn + fp) if (tn + fp) else 0

    return {
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "accuracy"   : round(accuracy, 4),
        "precision"  : round(precision, 4),
        "recall"     : round(recall, 4),
        "f1"         : round(f1, 4),
        "specificity": round(specificity, 4),
    }


def contingency_table(pairs: list[dict]) -> np.ndarray:
    """Filas: human (0=no-anotado, 1=anotado). Columnas: Qwen mascara (BUENA/PARCIAL/MALA)."""
    categories = ["BUENA", "PARCIAL", "MALA"]
    table = np.zeros((2, 3), dtype=int)
    for p in pairs:
        row = 1 if p["human_annotated"] else 0
        col = categories.index(p["qwen_mascara"]) if p["qwen_mascara"] in categories else 2
        table[row, col] += 1
    return table


# ---------------------------------------------------------------------------
# Visualizacion
# ---------------------------------------------------------------------------
def plot_results(pairs: list[dict], table: np.ndarray, kappa: float, metrics: dict, out_dir: Path):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Concordancia: Etiquetas Manuales vs Qwen2.5-VL", fontsize=13, fontweight="bold", y=1.02)

    # --- Panel 1: Tabla de contingencia ---
    ax = axes[0]
    im = ax.imshow(table, cmap="Blues", aspect="auto")
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["BUENA", "PARCIAL", "MALA"], fontsize=10)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Sin anotacion\n(human)", "Con anotacion\n(human)"], fontsize=9)
    ax.set_xlabel("Qwen2.5-VL: calidad de mascara", fontsize=10)
    ax.set_title("Tabla de contingencia\n(human anotado vs Qwen mascara)", fontsize=10)
    total = table.sum()
    for i in range(2):
        for j in range(3):
            v = table[i, j]
            pct = v / total * 100
            color = "white" if v > table.max() * 0.6 else "black"
            ax.text(j, i, f"{v}\n({pct:.1f}%)", ha="center", va="center",
                    fontsize=10, color=color, fontweight="bold")

    # --- Panel 2: Distribucion Qwen estados en subset anotado ---
    ax = axes[1]
    annotated  = [p for p in pairs if p["human_annotated"]]
    unannotated = [p for p in pairs if not p["human_annotated"]]
    estados = ["SANO", "ENFERMO", "DESCARTAR"]
    ann_counts  = [sum(1 for p in annotated   if p["qwen_estado"] == e) for e in estados]
    unann_counts = [sum(1 for p in unannotated if p["qwen_estado"] == e) for e in estados]
    x = np.arange(len(estados))
    w = 0.35
    b1 = ax.bar(x - w/2, ann_counts,  w, label="Con anotacion", color="#2980b9", alpha=0.85)
    b2 = ax.bar(x + w/2, unann_counts, w, label="Sin anotacion", color="#7f8c8d", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(estados, fontsize=10)
    ax.set_title("Estado Qwen segun decision humana", fontsize=10)
    ax.set_ylabel("N arboles", fontsize=9)
    ax.legend(fontsize=9)
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    for bar in list(b1) + list(b2):
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.5, str(int(h)),
                    ha="center", va="bottom", fontsize=8)

    # --- Panel 3: Metricas binarias ---
    ax = axes[2]
    ax.axis("off")
    kappa_interp = (
        "Casi perfecto (>0.80)" if kappa > 0.80 else
        "Sustancial (0.61-0.80)" if kappa > 0.60 else
        "Moderado (0.41-0.60)"   if kappa > 0.40 else
        "Aceptable (0.21-0.40)"  if kappa > 0.20 else
        "Leve (<0.20)"
    )
    lines = [
        ("Muestra total",          f"{len(pairs)} arboles"),
        ("Con anotacion (human)",  f"{len(annotated)} ({len(annotated)/len(pairs)*100:.1f}%)"),
        ("Sin anotacion (human)",  f"{len(unannotated)} ({len(unannotated)/len(pairs)*100:.1f}%)"),
        ("", ""),
        ("--- Binario: anotado vs BUENA ---", ""),
        ("Kappa de Cohen",         f"{kappa:.4f}  [{kappa_interp}]"),
        ("Exactitud",              f"{metrics['accuracy']:.4f}"),
        ("Precision",              f"{metrics['precision']:.4f}"),
        ("Recall (sensibilidad)",  f"{metrics['recall']:.4f}"),
        ("Especificidad",          f"{metrics['specificity']:.4f}"),
        ("F1-score",               f"{metrics['f1']:.4f}"),
        ("", ""),
        ("VP  (ann & BUENA)",      str(metrics["tp"])),
        ("VN  (!ann & !BUENA)",    str(metrics["tn"])),
        ("FP  (!ann & BUENA)",     str(metrics["fp"])),
        ("FN  (ann & !BUENA)",     str(metrics["fn"])),
    ]
    y_pos = 0.97
    for label, val in lines:
        if label.startswith("---"):
            ax.text(0.05, y_pos, label, transform=ax.transAxes, fontsize=9,
                    fontstyle="italic", color="#555555")
        else:
            ax.text(0.05, y_pos, label,  transform=ax.transAxes, fontsize=9, color="#222222")
            ax.text(0.62, y_pos, val,    transform=ax.transAxes, fontsize=9, color="#1a5276", fontweight="bold")
        y_pos -= 0.062

    plt.tight_layout()
    out_path = out_dir / "agreement_matrix.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Figura guardada: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Cargando datos...")
    coco, qwen = load_data()

    print(f"  COCO: {len(coco['images'])} imagenes, {len(coco['annotations'])} anotaciones")
    print(f"  Qwen: {len(qwen)} clasificaciones")

    pairs = build_pairs(coco, qwen)
    print(f"  Pares coincidentes: {len(pairs)}")

    # Binario: humano anotado (1) vs Qwen BUENA (1)
    y_human = [1 if p["human_annotated"]      else 0 for p in pairs]
    y_qwen  = [1 if p["qwen_mascara"] == "BUENA" else 0 for p in pairs]

    kappa   = cohens_kappa(y_human, y_qwen)
    metrics = binary_metrics(y_human, y_qwen)
    table   = contingency_table(pairs)

    # Distribuciones complementarias
    mascara_dist   = Counter(p["qwen_mascara"]   for p in pairs)
    estado_dist    = Counter(p["qwen_estado"]    for p in pairs)
    ann_estado     = Counter(p["qwen_estado"]    for p in pairs if p["human_annotated"])
    unann_mascara  = Counter(p["qwen_mascara"]   for p in pairs if not p["human_annotated"])

    # Impresion en consola
    print()
    print("=" * 60)
    print("  VALIDACION DE CONCORDANCIA — Etiquetas Manuales vs Qwen")
    print("=" * 60)
    print()
    print("Nota: La exportacion COCO contiene segmentacion de copas.")
    print("      La concordancia se mide en calidad de mascara:")
    print("      Human=anotado <-> Qwen=BUENA")
    print()
    print(f"Muestra:  {len(pairs)} arboles (de 450 revisados manualmente)")
    print()
    print("  Tabla de contingencia (filas=human, columnas=Qwen):")
    print(f"  {'':20s} {'BUENA':>8s} {'PARCIAL':>8s} {'MALA':>8s}  TOTAL")
    print(f"  {'Con anotacion':20s} {table[1,0]:>8d} {table[1,1]:>8d} {table[1,2]:>8d}  {table[1].sum():>5d}")
    print(f"  {'Sin anotacion':20s} {table[0,0]:>8d} {table[0,1]:>8d} {table[0,2]:>8d}  {table[0].sum():>5d}")
    print()
    print("  Metricas binarias (anotado vs BUENA):")
    print(f"    Kappa de Cohen  : {kappa:.4f}")
    print(f"    Exactitud       : {metrics['accuracy']:.4f}")
    print(f"    Precision       : {metrics['precision']:.4f}")
    print(f"    Recall          : {metrics['recall']:.4f}")
    print(f"    F1-score        : {metrics['f1']:.4f}")
    print(f"    Especificidad   : {metrics['specificity']:.4f}")
    print()

    # Veredicto
    if kappa > 0.60:
        verdict = "CONCORDANCIA SUSTANCIAL — Qwen valida con criterio humano"
    elif kappa > 0.40:
        verdict = "CONCORDANCIA MODERADA — Qwen aproxima criterio humano"
    elif kappa > 0.20:
        verdict = "CONCORDANCIA ACEPTABLE — sesgo leve detectable"
    else:
        verdict = "CONCORDANCIA BAJA — posible sesgo sistematico"

    symbol = "OK" if kappa > 0.40 else "AVISO"
    print(f"  [{symbol}] {verdict}")
    print()

    # Estado Qwen en subset anotado
    print("  Distribucion de estado Qwen en arboles con anotacion humana:")
    total_ann = len([p for p in pairs if p["human_annotated"]])
    for est in ["SANO", "ENFERMO", "DESCARTAR"]:
        n = ann_estado.get(est, 0)
        pct = n / total_ann * 100 if total_ann else 0
        print(f"    {est:12s}: {n:4d}  ({pct:.1f}%)")

    print()
    print("  Distribucion Qwen mascara en arboles SIN anotacion humana:")
    total_unann = len([p for p in pairs if not p["human_annotated"]])
    for m in ["BUENA", "PARCIAL", "MALA"]:
        n = unann_mascara.get(m, 0)
        pct = n / total_unann * 100 if total_unann else 0
        print(f"    {m:12s}: {n:4d}  ({pct:.1f}%)")

    print()
    print("Generando figura...")
    plot_results(pairs, table, kappa, metrics, OUT_DIR)

    # Guardar JSON
    result = {
        "muestra_total"          : len(pairs),
        "con_anotacion"          : int(sum(y_human)),
        "sin_anotacion"          : int(len(y_human) - sum(y_human)),
        "kappa"                  : round(kappa, 6),
        "veredicto"              : verdict,
        "metricas_binarias"      : metrics,
        "tabla_contingencia"     : {
            "filas"   : ["sin_anotacion", "con_anotacion"],
            "columnas": ["BUENA", "PARCIAL", "MALA"],
            "valores" : table.tolist(),
        },
        "dist_mascara_qwen"      : dict(mascara_dist),
        "dist_estado_qwen"       : dict(estado_dist),
        "dist_estado_anotados"   : dict(ann_estado),
        "dist_mascara_no_anotados": dict(unann_mascara),
    }
    out_json = OUT_DIR / "agreement_results.json"
    out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  JSON guardado: {out_json}")
    print()
    print("Listo.")


if __name__ == "__main__":
    main()
