@echo off
REM Run batch OCR extraction using GPU environment
REM Usage: run_batch_ocr.bat [options]

echo ========================================
echo Batch OCR Extraction (PaddleOCR GPU)
echo ========================================

REM Check if GPU environment exists
if not exist ".venv_paddle\Scripts\python.exe" (
    echo ERROR: GPU environment not found!
    echo Please run setup_paddle_env.bat first.
    pause
    exit /b 1
)

REM Default parameters
set INPUT_DIR=data\academic_dataset\classified_charts
set OUTPUT=data\cache\ocr_cache.json

REM Run batch OCR
echo.
echo Input: %INPUT_DIR%
echo Output: %OUTPUT%
echo.

.venv_paddle\Scripts\python.exe scripts\batch_ocr_gpu.py ^
    --input-dir %INPUT_DIR% ^
    --output %OUTPUT% ^
    --skip-existing ^
    --save-interval 500 ^
    %*

echo.
echo Done!
pause
