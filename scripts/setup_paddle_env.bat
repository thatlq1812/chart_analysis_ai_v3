@echo off
REM Setup separate Python environment for PaddleOCR GPU
REM This avoids conflicts with PyTorch CUDA in main environment

echo ========================================
echo Setting up PaddleOCR GPU Environment
echo ========================================

REM Create new virtual environment
echo [1/4] Creating virtual environment...
python -m venv .venv_paddle

REM Activate
echo [2/4] Activating environment...
call .venv_paddle\Scripts\activate.bat

REM Install PaddlePaddle GPU (CUDA 11.8)
echo [3/4] Installing PaddlePaddle GPU...
pip install paddlepaddle-gpu==2.6.2 -i https://pypi.tuna.tsinghua.edu.cn/simple

REM Install PaddleOCR and dependencies
echo [4/4] Installing PaddleOCR...
pip install paddleocr==2.10.0
pip install opencv-python-headless pillow numpy tqdm pyyaml

echo ========================================
echo Setup complete!
echo ========================================
echo.
echo To use: .venv_paddle\Scripts\python.exe scripts\batch_ocr_gpu.py
echo.
pause
