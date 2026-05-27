"""
server.py
Servidor FastAPI para la defensa del TT 2026-A127.
Sirve el reporte HTML y las imagenes de copas bajo demanda.

Uso:
  python server.py
  start_defensa.bat   (doble click)
"""
from __future__ import annotations

import io
import threading
import webbrowser
from functools import lru_cache
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, Response
from PIL import Image

# -- Rutas -------------------------------------------------------------------
BASE_DIR = Path(__file__).parent
CROPS_DIR   = BASE_DIR / "output" / "crops" / "rgb"
REPORT_HTML = BASE_DIR / "output" / "reporte_final" / "reporte_bosque_aragon.html"

PORT = 8000

# -- App ---------------------------------------------------------------------
app = FastAPI(title="Reporte Forestal TT 2026-A127", docs_url=None, redoc_url=None)
app.add_middleware(CORSMiddleware, allow_origins=["*"])

_PLACEHOLDER: bytes | None = None


def _make_placeholder() -> bytes:
    global _PLACEHOLDER
    if _PLACEHOLDER is None:
        img = Image.new("RGB", (200, 150), color=(220, 220, 220))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        _PLACEHOLDER = buf.getvalue()
    return _PLACEHOLDER


@lru_cache(maxsize=3000)
def _load_image(path_str: str) -> bytes | None:
    p = Path(path_str)
    return p.read_bytes() if p.exists() else None


@app.get("/")
def serve_report() -> FileResponse:
    if not REPORT_HTML.exists():
        raise HTTPException(status_code=404, detail="Reporte no generado aun. Ejecuta run_report.py")
    return FileResponse(REPORT_HTML, media_type="text/html")


@app.get("/imagen/{arbol_id}")
def serve_imagen(arbol_id: str) -> Response:
    # Evitar path traversal
    if "/" in arbol_id or "\\" in arbol_id or ".." in arbol_id:
        raise HTTPException(status_code=400, detail="ID invalido")

    data = _load_image(str(CROPS_DIR / f"{arbol_id}.png"))
    if data:
        return Response(content=data, media_type="image/png",
                        headers={"Cache-Control": "max-age=3600"})

    # Intentar .jpg si no existe .png
    data = _load_image(str(CROPS_DIR / f"{arbol_id}.jpg"))
    if data:
        return Response(content=data, media_type="image/jpeg",
                        headers={"Cache-Control": "max-age=3600"})

    return Response(content=_make_placeholder(), media_type="image/png",
                    headers={"Cache-Control": "max-age=3600"})


@app.get("/health")
def health() -> dict:
    crops_count = len(list(CROPS_DIR.glob("*.png"))) + len(list(CROPS_DIR.glob("*.jpg"))) \
                  if CROPS_DIR.exists() else 0
    return {
        "status": "ok",
        "reporte": REPORT_HTML.exists(),
        "crops_dir": str(CROPS_DIR),
        "crops_count": crops_count,
    }


# -- Arranque ----------------------------------------------------------------

def _print_banner() -> None:
    import socket
    try:
        hostname = socket.gethostname()
        local_ip = socket.gethostbyname(hostname)
    except Exception:
        local_ip = "?.?.?.?"

    print("=" * 50)
    print("  SERVIDOR DEFENSA - BOSQUE DE ARAGON")
    print("  TT 2026-A127  IPN ESCOM")
    print("=" * 50)
    print(f"  Esta PC  :  http://localhost:{PORT}")
    print(f"  Red local:  http://{local_ip}:{PORT}")
    print(f"  Tailscale:  (misma IP Tailscale):{PORT}")
    print("-" * 50)
    print("  -> Abre esa URL en tu laptop")
    print("  -> Ctrl+C para detener")
    print("=" * 50)


if __name__ == "__main__":
    _print_banner()
    threading.Timer(1.5, lambda: webbrowser.open(f"http://localhost:{PORT}")).start()
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="warning")
