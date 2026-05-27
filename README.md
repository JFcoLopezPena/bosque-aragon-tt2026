# Prototipo de Priorizacion Forestal
## Bosque de San Juan de Aragon, CDMX

[![GitHub Pages](https://img.shields.io/badge/Reporte-GitHub%20Pages-blue)](https://JFcoLopezPena.github.io/bosque-aragon-tt2026)
[![IPN](https://img.shields.io/badge/Institucion-ESCOM%20IPN-darkred)](https://www.escom.ipn.mx)
[![TT](https://img.shields.io/badge/TT-2026--A127-orange)]()
[![Python](https://img.shields.io/badge/Python-3.13-blue)]()

Prototipo basado en vision por computadora para la jerarquizacion y priorizacion de areas de atencion forestal en el Bosque de San Juan de Aragon (162 ha, CDMX), mediante el procesamiento de ortomosaicos multiespectrales captados con dron DJI Mavic 3 Multispectral y fine-tuning del modelo fundacional SAM (Segment Anything Model).

## Resultados principales

| Metrica | Valor |
|---|---|
| Arboles detectados y georreferenciados | 15,642 |
| Tiempo de procesamiento (162 ha) | 58.8 minutos |
| IoU segmentacion — linea base SAM | 0.615 |
| IoU segmentacion — fine-tuning R1 | 0.881 |
| F1-score macro (clasificacion) | 0.747 |
| Accuracy global | 75.3% |
| Recall estres severo | 80.1% |

## Reporte interactivo

Ver el mapa interactivo en:
**https://JFcoLopezPena.github.io/bosque-aragon-tt2026**

## Arquitectura del pipeline

```
Ortomosaico multiespectral (39,899 x 33,480 px, 5 cm/px)
        |
[01] Pipeline de deteccion SAM
     -> 1,710 tiles | 15,642 arboles detectados
        |
[02] Clasificacion automatica con Qwen2.5-VL
     -> Validacion de calidad de mascaras: 18.8% -> 79.8% mascara BUENA
        |
[03] Fine-tuning SAM — Ronda 1 (Segmentacion)
     -> IoU: 0.615 -> 0.881
        |
[04] Fine-tuning SAM — Ronda 2 (Clasificacion)
     -> F1: 0.747 | Accuracy: 0.753
        |
[05] Reporte interactivo FastAPI + Leaflet.js
     -> Mapa georreferenciado + ranking de zonas prioritarias
```

## Instalacion

```bash
git clone https://github.com/JFcoLopezPena/bosque-aragon-tt2026
cd bosque-aragon-tt2026
pip install -r requirements.txt

# Descargar checkpoint SAM ViT-B (~375 MB)
mkdir checkpoints
curl -L -o checkpoints/sam_vit_b_01ec64.pth \
  https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
```

## Uso

### 1. Pipeline de deteccion
```bash
cd 01_pipeline_deteccion
python run_all.py
```

### 2. Clasificacion con Qwen (requiere Ollama)
```bash
ollama pull qwen2.5vl:7b-q8_0
cd 02_clasificacion_qwen
python classify_trees_local.py --pilot
python classify_trees_local.py
```

### 3. Fine-tuning Ronda 1 — Segmentacion
```bash
cd 03_finetune_segmentacion
python train.py
python evaluate.py
```

### 4. Fine-tuning Ronda 2 — Clasificacion
```bash
cd 04_finetune_clasificacion
python train.py
python evaluate.py
```

### 5. Reporte interactivo
```bash
cd 05_reporte_interactivo
python server.py
# Abrir: http://localhost:8000
```
En Windows: doble click en `start_defensa.bat`

## Requisitos de hardware

- GPU NVIDIA con minimo 8 GB VRAM (recomendado 16 GB)
- 16 GB RAM minimo (recomendado 64 GB para Qwen2.5-VL)
- 100 GB de espacio en disco para datos y outputs

## Estructura del repositorio

```
bosque-aragon-tt2026/
|-- docs/                        # Reporte estatico GitHub Pages
|-- 01_pipeline_deteccion/       # Deteccion SAM + georreferenciacion
|-- 02_clasificacion_qwen/       # Clasificacion VLM con Qwen2.5-VL
|-- 03_finetune_segmentacion/    # Fine-tuning SAM Ronda 1
|-- 04_finetune_clasificacion/   # Fine-tuning SAM Ronda 2
|-- 05_reporte_interactivo/      # Servidor FastAPI + Leaflet
|-- requirements.txt
`-- README.md
```

## Autores

**Jose Francisco Lopez Pena** — jlopezp2102@alumno.ipn.mx  
**Juan Manuel Venegas Salinas** — jvenegass1400@alumno.ipn.mx

**Directores:**  
Dr. Marco Antonio Moreno Ibarra — CIC-IPN  
Dr. Roberto Zagal Flores — ESCOM-IPN

Escuela Superior de Computo — Instituto Politecnico Nacional  
Licenciatura en Ciencia de Datos | Trabajo Terminal No. 2026-A127

## Licencia

MIT License
