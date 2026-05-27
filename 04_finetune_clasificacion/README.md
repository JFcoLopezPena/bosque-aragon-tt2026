# 04 — Fine-tuning SAM Ronda 2 (Clasificacion)

Fine-tuning supervisado de SAM ViT-B para clasificacion binaria de copas (con atencion / sin atencion). Arquitectura dual: head de clasificacion (GAP + MLP) + decoder de segmentacion heredado de R1. Loss combinada 0.6 cls + 0.4 seg.

## Archivos

| Archivo | Funcion |
|---|---|
| `config.yaml` | Hiperparametros, pesos de clase, rutas |
| `dataset.py` | Dataset con splits estratificados (70/15/15) |
| `model.py` | SAM ViT-B + ClassificationHead (AdaptiveAvgPool + MLP) |
| `losses.py` | Focal + Dice para segmentacion; Focal + CE para clasificacion |
| `train.py` | Entrenamiento con AMP, early stopping en F1-atencion |
| `evaluate.py` | Matriz de confusion, precision/recall/F1 por clase, IoU |

## Parametros clave (config.yaml)

```yaml
training:
  batch_size: 4
  learning_rate: 1.0e-5
  num_epochs: 100
  early_stopping_patience: 15
  image_size: 256
  mixed_precision: true
  class_weights: [1.0, 1.68]   # inverso frecuencia: sano=62.7%, atencion=37.3%

loss:
  cls_weight: 0.6
  seg_weight: 0.4
  focal_gamma: 2.0
```

## Ejecucion

```bash
python prepare_dataset_r2.py    # desde raiz del proyecto
python train.py
python evaluate.py

# Reanudar desde checkpoint:
python train.py --resume
```

## Salida esperada

```
Dataset: ~9,900 arboles (mascara BUENA)
  Train: 6,930 | Val: 1,485 | Test: 1,485

Metricas en test set:
  F1-macro:       0.747
  Accuracy:       0.753
  F1-atencion:    0.707
  Recall-atencion: 0.801
  IoU segmentacion: 0.615

Checkpoint: checkpoints/cls_finetuned/best_model.pth
```
