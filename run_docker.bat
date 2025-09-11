@echo off
setlocal enabledelayedexpansion

REM Настройки
set IMAGE_NAME=t-bank-logo
set PORT=8000

echo.
echo ==============================================================
echo               DOCKER RUN: %IMAGE_NAME%              
echo ==============================================================
echo.

REM Проверяем доступность GPU
echo Checking GPU availability...
echo.

set GPU_AVAILABLE=false

REM Способ 1: Проверяем через nvidia-smi
where nvidia-smi >nul 2>nul
if !errorlevel! equ 0 (
    nvidia-smi >nul 2>nul
    if !errorlevel! equ 0 (
        set GPU_AVAILABLE=true
        echo NVIDIA GPU detected via nvidia-smi
    )
)

REM Способ 2: Проверяем через docker (если первый способ не сработал)
if "!GPU_AVAILABLE!"=="false" (
    echo Checking via Docker...
    docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi >nul 2>nul
    if !errorlevel! equ 0 (
        set GPU_AVAILABLE=true
        echo GPU is available via Docker
    )
)

REM Способ 3: Проверяем наличие драйверов в системе
if "!GPU_AVAILABLE!"=="false" (
    where nvcc >nul 2>nul
    if !errorlevel! equ 0 (
        set GPU_AVAILABLE=true
        echo NVIDIA CUDA Discovered
    )
)

if "!GPU_AVAILABLE!"=="true" (
    echo.
    echo GPU AVAILABLE! Launching with GPU support.
    echo.
    
    REM Запускаем с GPU
    echo Launch: docker run -p %PORT%:%PORT% --gpus all %IMAGE_NAME%
    echo.
    docker run -p %PORT%:%PORT% --gpus all %IMAGE_NAME%
    
    set RUN_RESULT=!errorlevel!
    set RUN_TYPE=GPU
) else (
    echo.
    echo GPU not available. Running on CPU...
    echo.
    
    REM Запускаем на CPU
    echo Launch: docker run -p %PORT%:%PORT% %IMAGE_NAME%
    echo.
    docker run -p %PORT%:%PORT% %IMAGE_NAME%
    
    set RUN_RESULT=!errorlevel!
    set RUN_TYPE=CPU
)

REM Обработка результата запуска
echo.
echo ==============================================================

if !RUN_RESULT! equ 0 (
    echo The container completed its work successfully.
) else (
    echo The container exited with an error: !RUN_RESULT!
)

echo Operating mode: !RUN_TYPE!
echo Port: %PORT%
echo.
echo To stop the container, press Ctrl+C
echo.

pause