@echo off
title Servidor Defensa TT 2026-A127 - IPN ESCOM
cd /d C:\Users\fcolo\Desktop\TT\reporte_pipeline

echo Instalando dependencias...
pip install fastapi uvicorn pillow --quiet

echo.
echo Iniciando servidor...
python server.py

pause
