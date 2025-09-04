@echo off
setlocal enabledelayedexpansion

REM Настройки
set IMAGE_NAME=t-bank-logo
set DOCKER_BUILDKIT=1

REM Цвета для красивого вывода
for /F "tokens=1,2 delims=#" %%a in ('"prompt #$H#$E# & echo on & for %%b in (1) do rem"') do (
  set "DEL=%%a"
)

echo.
echo ==============================================================
echo                    DOCKER BUILD STARTED                      
echo ==============================================================
echo.
echo [%DATE% %TIME%] Build the image: %IMAGE_NAME%
echo.

REM Засекаем время начала
set start_time=%TIME%

REM Функция для вывода шагов с нумерацией
set /a step_counter=0

REM Выполняем сборку с обработкой вывода построчно
docker build --progress=plain -t %IMAGE_NAME% .

REM Проверяем результат сборки
if !errorlevel! equ 0 (
    echo.
    echo ==============================================================
    echo                       BUILD SUCCESSFUL                      
    echo ==============================================================
    echo.
    echo The build was completed successfully!
    echo Lead time: !start_time! - %TIME%
    echo.
    echo Image information:
    docker images %IMAGE_NAME% --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}\t{{.CreatedSince}}"
) else (
    echo.
    echo ==============================================================
    echo                        BUILD FAILED                      
    echo ==============================================================
    echo.
    echo Build error! Code: !errorlevel!
    echo Lead time: !start_time! - %TIME%
    echo.
    echo For details, run: docker build -t %IMAGE_NAME% . --progress=plain
)

echo.
pause
endlocal