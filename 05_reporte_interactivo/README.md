# 05 — Reporte Interactivo

Servidor FastAPI que sirve el reporte HTML con mapa Leaflet, graficas Chart.js y las imagenes aereas de cada copa. Accesible desde red local y via Tailscale para presentaciones remotas.

## Archivos

| Archivo | Funcion |
|---|---|
| `server.py` | FastAPI: sirve HTML + imagenes de copas bajo demanda |
| `start_defensa.bat` | Arranque con doble click en Windows |

## Prerequisitos

Antes de arrancar el servidor, generar el reporte ejecutando desde la raiz:

```bash
python reporte_pipeline/run_report.py
# Genera: output/reporte_final/reporte_bosque_aragon.html
```

## Ejecucion

```bash
pip install fastapi uvicorn pillow
python server.py
```

O en Windows: doble click en `start_defensa.bat`

## Endpoints

| Endpoint | Descripcion |
|---|---|
| `GET /` | Reporte HTML completo |
| `GET /imagen/{arbol_id}` | Imagen RGB de la copa (con cache LRU) |
| `GET /health` | Estado del servidor y conteo de crops |

## Acceso

```
Esta PC:    http://localhost:8000
Red local:  http://[IP-local]:8000
Tailscale:  http://[IP-Tailscale]:8000
```

## Reporte estatico (sin servidor)

La version para GitHub Pages esta en `docs/index.html` y abre
directamente en el navegador sin ninguna dependencia de servidor.
Ver: https://JFcoLopezPena.github.io/bosque-aragon-tt2026
