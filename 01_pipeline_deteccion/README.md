# 01 — Pipeline de Deteccion SAM

Procesa el ortomosaico multiespectral (39,899 x 33,480 px) para detectar y georreferenciar las copas de los arboles usando SAM ViT-B con prompts automaticos por cuadricula. Genera un GeoJSON con 15,642 arboles y sus coordenadas WGS84.

## Archivos

| Archivo | Funcion |
|---|---|
| `config.yaml` | Parametros globales (tile size, umbrales, rutas) |
| `01_align.py` | Alineacion de bandas RGB y NIR del ortomosaico |
| `02_tiling.py` | Division en 1,710 tiles de 512x512 px con 10% solapamiento |
| `03_sam_inference.py` | Inferencia SAM con prompts por cuadricula (32x32) |
| `04_filter.py` | Filtrado de mascaras por area, forma e IoU |
| `05_crop.py` | Extraccion de crops RGB e imagenes de mascara por arbol |
| `06_export.py` | Exportacion a GeoJSON con coordenadas UTM->WGS84 |
| `run_all.py` | Orquestador: ejecuta pasos 01-06 en secuencia |

## Parametros clave (config.yaml)

```yaml
tile_size: 512          # pixeles por tile
overlap: 0.10           # solapamiento entre tiles
sam_model: vit_b        # modelo SAM
points_per_side: 32     # densidad de prompts por tile
min_area_px: 200        # area minima de copa en pixeles
max_area_px: 50000      # area maxima de copa en pixeles
```

## Ejecucion

```bash
# Requiere checkpoint SAM en: checkpoints/sam_vit_b_01ec64.pth
python run_all.py
```

## Salida esperada

```
output/
|-- metadata/detections.geojson    # 15,642 arboles georreferenciados
|-- crops/rgb/                     # ~15,642 imagenes RGB de copas
|-- masks_r1/                      # ~15,642 mascaras binarias
`-- tiles/                         # 1,710 tiles intermedios

Tiempo total: ~58.8 min (GPU RTX 3080 16 GB)
```
