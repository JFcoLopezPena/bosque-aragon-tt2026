# 03 — Fine-tuning SAM Ronda 1 (Segmentacion)

Fine-tuning supervisado del mask decoder de SAM ViT-B para mejorar la segmentacion de copas de arboles. Usa las mascaras BUENA de la clasificacion Qwen como ground truth. Mejora el IoU de 0.615 (SAM base) a 0.881.

## Archivos

| Archivo | Funcion |
|---|---|
| `config.yaml` | Hiperparametros de entrenamiento y rutas |
| `dataset.py` | Dataset PyTorch con augmentaciones (flip, rotacion, elastic) |
| `model.py` | SAM ViT-B con decoder entrenable y encoder congelado |
| `losses.py` | Loss combinada: Focal (0.5) + Dice (0.5) con ROI masking |
| `train.py` | Bucle de entrenamiento con early stopping y checkpointing |
| `evaluate.py` | Metricas en test set: IoU, Dice, visualizaciones |
| `predict.py` | Inferencia sobre nuevas imagenes |

## Parametros clave (config.yaml)

```yaml
training:
  batch_size: 4
  learning_rate: 1.0e-4
  num_epochs: 100
  early_stopping_patience: 15
  image_size: 512
  mixed_precision: true

loss:
  focal_weight: 0.5
  dice_weight: 0.5
  use_roi_mask: true
  roi_dilation_px: 10
```

## Ejecucion

```bash
python train.py
python evaluate.py
# Checkpoint: checkpoints/seg_finetuned/best_model.pth
```

## Salida esperada

```
Entrenamiento: ~45 min/epoca (GPU RTX 3080 16 GB)
Mejor modelo: epoch ~40-60
IoU test: 0.881 (vs 0.615 SAM base)
Dice test: 0.927
```
