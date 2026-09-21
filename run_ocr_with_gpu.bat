@echo off
REM OCR Benchmark with GPU Support
REM This script sets up CUDA/cuDNN before running Streamlit

echo.
echo ============================================
echo OCR Benchmark - GPU Configuration
echo ============================================
echo.

REM Set CUDA paths
set CUDA_PATH=C:\Program Files\NVIDIA\CUDNN\v9.26
set CUDNN_PATH=C:\Program Files\NVIDIA\CUDNN\v9.26\bin\13.4\x64
set CUDA_HOME=C:\Program Files\NVIDIA\CUDNN\v9.26

REM Add to PATH
set PATH=%CUDNN_PATH%;%CUDA_PATH%\bin;%CUDA_PATH%\lib\13.4\x64;%PATH%

REM Enable GPU for Paddle
set PADDLE_ENABLE_GPU=1

echo ✓ CUDA_PATH = %CUDA_PATH%
echo ✓ CUDNN_PATH = %CUDNN_PATH%
echo ✓ PADDLE_ENABLE_GPU = %PADDLE_ENABLE_GPU%
echo.
echo Starting Streamlit app...
echo.

REM Activate your virtual environment (if needed)
REM call .venv\Scripts\activate

REM Run streamlit
streamlit run app.py

pause