@echo off
cd /d "%~dp0"
python -m echog smoke --out outputs/smoke
pause
