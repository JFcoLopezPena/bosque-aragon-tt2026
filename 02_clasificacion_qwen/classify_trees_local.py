"""
classify_trees_local.py
Clasifica pares de imágenes de copas de árboles usando Qwen2.5-VL-7B via Ollama.
Tesis: detección de enfermedades en árboles, Bosque de Aragón CDMX
"""

import argparse
import base64
import gc
import json
import sys
import time
from pathlib import Path

import requests
from tqdm import tqdm

# ── Rutas ──────────────────────────────────────────────────────────────────────
BASE_DIR  = Path(__file__).parent.parent
RGB_DIR   = BASE_DIR / "output" / "crops" / "rgb"
MASK_DIR  = BASE_DIR / "output" / "masks_r1"
OUT_FILE  = BASE_DIR / "output" / "clasificaciones_r1.json"
PREV_FILE = BASE_DIR / "output" / "clasificaciones_local.json"

# ── Configuración Ollama ────────────────────────────────────────────────────────
OLLAMA_URL  = "http://localhost:11434/api/chat"
OLLAMA_PING = "http://localhost:11434/api/tags"
MODEL       = "qwen2.5vl:7b-q8_0"

BATCH_SIZE      = 3
CHECKPOINT_FREQ = 100
MAX_RETRIES     = 2
PILOT_COUNT     = 20

SYSTEM_PROMPT = (
    "Eres un experto en detección de enfermedades en árboles urbanos desde "
    "imágenes aéreas de dron multiespectral. Recibirás DOS imágenes por árbol:\n"
    "1. Crop RGB de la copa del árbol (imagen a color)\n"
    "2. Máscara binaria generada por modelo SAM fine-tuneado (mayor precisión que SAM automático; "
    "blanco=copa detectada, negro=fondo)\n\n"
    "Tu tarea es evaluar DOS cosas:\n\n"
    "MÁSCARA: ¿La máscara binaria cubre correctamente la copa?\n"
    "- BUENA: coincide bien con la copa visible en el RGB (bordes ajustados, sin exceso de fondo)\n"
    "- PARCIAL: cubre solo parte de la copa o incluye área significativa de fondo/otros árboles\n"
    "- MALA: falso positivo, no corresponde a ningún árbol, o la región está muy desviada\n\n"
    "ESTADO: ¿El árbol está sano o enfermo? Usa principalmente la región de la máscara para el análisis.\n"
    "- SANO: copa densa, verde uniforme, sin huecos visibles, forma redondeada o compacta\n"
    "- ENFERMO: defoliación visible, color café/amarillo/grisáceo, huecos internos en copa, forma asimétrica o irregular\n"
    "- DESCARTAR: sombra excesiva que impide evaluar, imagen borrosa o muy pequeña, máscara MALA que hace inválido el análisis\n\n"
    'Responde ÚNICAMENTE con este JSON, sin texto adicional, sin explicaciones:\n'
    '{"mascara": "BUENA/PARCIAL/MALA", "estado": "SANO/ENFERMO/DESCARTAR"}'
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def check_ollama() -> None:
    """Verifica que Ollama está corriendo; sale con error claro si no."""
    try:
        r = requests.get(OLLAMA_PING, timeout=5)
        r.raise_for_status()
    except requests.exceptions.ConnectionError:
        print(
            "\n[ERROR] No se pudo conectar a Ollama en http://localhost:11434\n"
            "        Asegúrate de que Ollama está corriendo: ollama serve\n"
            "        Y que el modelo está disponible: ollama pull qwen2.5vl:7b-q8_0"
        )
        sys.exit(1)
    except requests.exceptions.RequestException as e:
        print(f"\n[ERROR] Problema al verificar Ollama: {e}")
        sys.exit(1)


def encode_image(path: Path) -> str:
    """Lee un archivo de imagen y lo convierte a base64."""
    return base64.b64encode(path.read_bytes()).decode("utf-8")


def build_pairs() -> list[tuple[str, Path, Path]]:
    """
    Construye la lista de (tree_id, rgb_path, mask_path).
    Sale si falta alguna máscara para un RGB existente.
    """
    rgb_files = sorted(RGB_DIR.glob("*.png")) + sorted(RGB_DIR.glob("*.jpg"))
    if not rgb_files:
        print(f"[ERROR] No se encontraron imágenes RGB en {RGB_DIR}")
        sys.exit(1)

    pairs: list[tuple[str, Path, Path]] = []
    missing: list[str] = []

    for rgb_path in rgb_files:
        tree_id = rgb_path.stem

        # Buscar máscara con mismo stem (.png o .jpg)
        mask_path: Path | None = None
        for ext in (".png", ".jpg", ".jpeg"):
            candidate = MASK_DIR / (tree_id + ext)
            if candidate.exists():
                mask_path = candidate
                break

        if mask_path is None:
            missing.append(tree_id)
        else:
            pairs.append((tree_id, rgb_path, mask_path))

    if missing:
        print(f"[WARN] {len(missing)} árbol(es) sin máscara R1, se omitirán:")
        for m in missing[:10]:
            print(f"   - {m}")
        if len(missing) > 10:
            print(f"   ... y {len(missing) - 10} más")

    return pairs


def load_checkpoint() -> dict:
    """Carga clasificaciones existentes para reanudar sin reprocesar."""
    if OUT_FILE.exists():
        try:
            data = json.loads(OUT_FILE.read_text(encoding="utf-8"))
            print(f"[INFO] Checkpoint cargado: {len(data)} árboles ya clasificados")
            return data
        except json.JSONDecodeError:
            print("[WARN] Checkpoint corrupto, se ignorará y se empezará de cero")
    return {}


def save_checkpoint(results: dict) -> None:
    """Guarda el diccionario de resultados en disco."""
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def classify_tree(
    tree_id: str,
    rgb_path: Path,
    mask_path: Path,
    attempt: int = 0,
    debug: bool = False,
) -> dict:
    """
    Envía el par de imágenes a Ollama y devuelve el dict con mascara/estado.
    Reintenta hasta MAX_RETRIES veces si el JSON viene malformado.
    """
    rgb_b64  = encode_image(rgb_path)
    mask_b64 = encode_image(mask_path)

    # Ollama espera las imágenes en el campo "images" del mensaje de usuario,
    # no en content como image_url (eso es formato OpenAI).
    payload = {
        "model": MODEL,
        "stream": False,
        "options": {"temperature": 0},
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Árbol ID: {tree_id}\n"
                    "Imagen 1: Crop RGB de la copa\n"
                    "Imagen 2: Máscara binaria fine-tuneada (SAM R1)\n"
                    "Evalúa máscara y estado del árbol."
                ),
                "images": [rgb_b64, mask_b64],
            },
        ],
    }

    try:
        r = requests.post(OLLAMA_URL, json=payload, timeout=120)
        r.raise_for_status()
        response_json = r.json()
        raw = response_json["message"]["content"].strip()

        if debug:
            print("\n" + "─" * 60)
            print(f"[DEBUG] Árbol: {tree_id}")
            print(f"[DEBUG] RGB:   {rgb_path}")
            print(f"[DEBUG] Mask:  {mask_path}")
            print("[DEBUG] ── Respuesta RAW del modelo ──────────────────────")
            print(raw)
            print("[DEBUG] ── Fin respuesta RAW ────────────────────────────")
            print(f"[DEBUG] Longitud respuesta: {len(raw)} caracteres")
            print("─" * 60 + "\n")

        # Extraer JSON aunque venga rodeado de texto extra
        start = raw.find("{")
        end   = raw.rfind("}") + 1
        if start == -1 or end == 0:
            if debug:
                print(f"[DEBUG] No se encontró '{{' ni '}}' en la respuesta")
            raise ValueError(f"No se encontró JSON en la respuesta: {raw[:200]}")

        json_str = raw[start:end]
        if debug:
            print(f"[DEBUG] Fragmento JSON extraído: {json_str}")

        parsed = json.loads(json_str)

        mascara = str(parsed.get("mascara", "ERROR")).upper()
        estado  = str(parsed.get("estado",  "ERROR")).upper()

        valid_mascara = {"BUENA", "PARCIAL", "MALA", "ERROR"}
        valid_estado  = {"SANO", "ENFERMO", "DESCARTAR", "ERROR"}

        if mascara not in valid_mascara:
            if debug:
                print(f"[DEBUG] Valor de mascara no válido: '{mascara}'")
            mascara = "ERROR"
        if estado not in valid_estado:
            if debug:
                print(f"[DEBUG] Valor de estado no válido: '{estado}'")
            estado = "ERROR"

        if debug:
            print(f"[DEBUG] Resultado final: mascara={mascara}, estado={estado}\n")

        return {"mascara": mascara, "estado": estado}

    except (ValueError, KeyError, json.JSONDecodeError) as e:
        if debug:
            print(f"[DEBUG] Error al parsear (intento {attempt + 1}): {e}\n")
        if attempt < MAX_RETRIES:
            return classify_tree(tree_id, rgb_path, mask_path, attempt + 1, debug)
        return {"mascara": "ERROR", "estado": "ERROR"}

    except requests.exceptions.RequestException as e:
        if debug:
            print(f"[DEBUG] Error de red (intento {attempt + 1}): {e}\n")
        if attempt < MAX_RETRIES:
            time.sleep(2)
            return classify_tree(tree_id, rgb_path, mask_path, attempt + 1, debug)
        return {"mascara": "ERROR", "estado": "ERROR"}


def print_summary(results: dict) -> None:
    """Imprime estadísticas finales y comparativa con clasificaciones anteriores."""
    total = len(results)
    sanos      = sum(1 for v in results.values() if v["estado"]  == "SANO")
    enfermos   = sum(1 for v in results.values() if v["estado"]  == "ENFERMO")
    descartar  = sum(1 for v in results.values() if v["estado"]  == "DESCARTAR")
    errores    = sum(1 for v in results.values() if v["estado"]  == "ERROR")
    m_buena    = sum(1 for v in results.values() if v["mascara"] == "BUENA")
    m_parcial  = sum(1 for v in results.values() if v["mascara"] == "PARCIAL")
    m_mala     = sum(1 for v in results.values() if v["mascara"] == "MALA")

    pct_buena = 100.0 * m_buena / max(total, 1)

    print("\n" + "=" * 52)
    print("  RESUMEN DE CLASIFICACION (R1 - SAM fine-tuneado)")
    print("=" * 52)
    print(f"  Total procesados : {total}")
    print(f"  Sanos            : {sanos}")
    print(f"  Enfermos         : {enfermos}")
    print(f"  Descartar        : {descartar}")
    print(f"  Errores          : {errores}")
    print("  ------------------------------------------------")
    print(f"  Mascaras BUENA   : {m_buena}  ({pct_buena:.1f}%)")
    print(f"  Mascaras PARCIAL : {m_parcial}")
    print(f"  Mascaras MALA    : {m_mala}")
    print("=" * 52)
    print(f"  Guardado en: {OUT_FILE}")

    # Comparativa con Ronda 0
    if PREV_FILE.exists():
        try:
            prev = json.loads(PREV_FILE.read_text(encoding="utf-8"))
            prev_total  = len(prev)
            prev_buena  = sum(1 for v in prev.values() if v.get("mascara") == "BUENA")
            prev_pct    = 100.0 * prev_buena / max(prev_total, 1)
            delta       = pct_buena - prev_pct

            print()
            print("  COMPARATIVA R0 (SAM auto) vs R1 (fine-tuneado)")
            print("  ------------------------------------------------")
            print(f"  R0 mascaras BUENA : {prev_buena}/{prev_total}  ({prev_pct:.1f}%)")
            print(f"  R1 mascaras BUENA : {m_buena}/{total}  ({pct_buena:.1f}%)")
            if delta > 2:
                print(f"  Mejora            : +{delta:.1f} pp  -- fine-tuning efectivo")
            elif delta < -2:
                print(f"  Retroceso         : {delta:.1f} pp  -- revisar threshold o datos")
                print("  Sugerencia        : prueba --threshold 0.60 en generate_masks_r1.py")
            else:
                print(f"  Delta             : {delta:+.1f} pp  (sin cambio significativo)")
                print("  Sugerencia        : revisa visualizaciones en output/masks_r1/")
        except (json.JSONDecodeError, KeyError):
            print("[WARN] No se pudo cargar el archivo de comparacion anterior.")

    print("=" * 52 + "\n")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Clasifica copas de árboles con Qwen2.5-VL-7B via Ollama"
    )
    parser.add_argument(
        "--pilot",
        action="store_true",
        help=f"Procesa solo los primeros {PILOT_COUNT} árboles para validar calidad",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Procesa 1 árbol e imprime la respuesta RAW del modelo para diagnóstico",
    )
    args = parser.parse_args()

    # 0. Validar directorio de máscaras R1
    if not MASK_DIR.exists():
        print(f"[ERROR] Directorio de máscaras R1 no existe: {MASK_DIR}")
        print("        Ejecuta primero: python generate_masks_r1.py")
        sys.exit(1)
    mask_count = len(list(MASK_DIR.glob("*.png")))
    if mask_count == 0:
        print(f"[ERROR] No se encontraron máscaras PNG en {MASK_DIR}")
        print("        Ejecuta primero: python generate_masks_r1.py")
        sys.exit(1)
    print(f"[INFO] Directorio de máscaras R1 OK: {mask_count} máscaras encontradas")

    # 1. Verificar Ollama
    print("[INFO] Verificando conexion con Ollama...")
    check_ollama()
    print(f"[INFO] Ollama OK -- modelo: {MODEL}")

    # 2. Construir pares
    print("[INFO] Construyendo lista de pares RGB+máscara...")
    pairs = build_pairs()

    if args.debug:
        pairs = pairs[:1]
        print(f"[INFO] Modo DEBUG: procesando 1 árbol con salida RAW del modelo")
    elif args.pilot:
        pairs = pairs[:PILOT_COUNT]
        print(f"[INFO] Modo PILOT: procesando los primeros {len(pairs)} árboles")
    else:
        print(f"[INFO] {len(pairs)} árboles encontrados")

    # 3. Cargar checkpoint (debug siempre reprocesa, ignora checkpoint)
    results = {} if args.debug else load_checkpoint()

    # Filtrar ya procesados
    pending = [(tid, r, m) for tid, r, m in pairs if tid not in results]
    print(f"[INFO] Pendientes: {len(pending)} (ya clasificados: {len(pairs) - len(pending)})")

    if not pending:
        print("[INFO] Nada que procesar — todos los árboles ya están clasificados.")
        print_summary(results)
        return

    # 4. Procesar en lotes
    start_time = time.time()
    processed_this_run = 0

    with tqdm(
        total=len(pending),
        desc="Clasificando",
        unit="árbol",
        dynamic_ncols=True,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
    ) as pbar:
        for i in range(0, len(pending), BATCH_SIZE):
            batch = pending[i : i + BATCH_SIZE]

            for tree_id, rgb_path, mask_path in batch:
                result = classify_tree(tree_id, rgb_path, mask_path, debug=args.debug)
                results[tree_id] = result
                processed_this_run += 1
                pbar.update(1)

                # Actualizar descripción con último resultado
                m = result["mascara"]
                e = result["estado"]
                pbar.set_postfix({"ultimo": f"{tree_id[-12:]} -> {m}/{e}"}, refresh=False)

            # Checkpoint periódico (no en modo debug para no contaminar el archivo)
            if not args.debug and processed_this_run % CHECKPOINT_FREQ == 0:
                save_checkpoint(results)

            # Limpieza de memoria
            gc.collect()

    # 5. Guardado final (no en modo debug)
    if not args.debug:
        save_checkpoint(results)
    elapsed = time.time() - start_time
    rate = processed_this_run / (elapsed / 60) if elapsed > 0 else 0
    print(f"\n[INFO] Completado en {elapsed / 60:.1f} min ({rate:.1f} árboles/min)")

    print_summary(results)


if __name__ == "__main__":
    main()
