# 02 — Clasificacion Automatica con Qwen2.5-VL

Clasifica la calidad de las mascaras de copas usando el modelo de vision Qwen2.5-VL-7B via Ollama. Determina si cada mascara es BUENA, PARCIAL o MALA para filtrar los datos de entrenamiento del fine-tuning. Mejora la tasa de mascaras BUENA del 18.8% al 79.8%.

## Archivos

| Archivo | Funcion |
|---|---|
| `classify_trees_local.py` | Clasificacion por lotes usando Ollama API local |

## Requisitos

```bash
# Instalar Ollama: https://ollama.com/download
ollama pull qwen2.5vl:7b-q8_0   # ~8 GB descarga
```

## Ejecucion

```bash
# Prueba con 10 arboles (pilot mode)
python classify_trees_local.py --pilot

# Clasificacion completa (~12,000 arboles)
python classify_trees_local.py
```

## Parametros clave

```python
BATCH_SIZE      = 3      # arboles por request a Ollama
CHECKPOINT_FREQ = 100    # guardar progreso cada N arboles
MODEL = "qwen2.5vl:7b-q8_0"
```

## Salida esperada

```
output/clasificaciones_r1.json
  {
    "tile_0001_0002": {"mascara": "BUENA",   "estado": "SANO"},
    "tile_0001_0003": {"mascara": "PARCIAL", "estado": "ENFERMO"},
    ...
  }

Tiempo: ~4-6 horas para 15,642 arboles (CPU i7-13700K)
Resultado: 79.8% mascaras BUENA (vs 18.8% linea base SAM)
```
