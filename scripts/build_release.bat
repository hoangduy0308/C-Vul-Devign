@echo off
REM Devign Scanner - Quick Build Script for Windows
REM Usage: build_release.bat v1.0.0

setlocal enabledelayedexpansion

set VERSION=%1
if "%VERSION%"=="" set VERSION=v1.0.0

echo ==========================================
echo Devign Scanner Release Builder
echo Version: %VERSION%
echo ==========================================

set ROOT_DIR=%~dp0..
set BUILD_DIR=%ROOT_DIR%\build
set RELEASE_DIR=%BUILD_DIR%\release

REM Clean previous build
if exist "%BUILD_DIR%" rmdir /s /q "%BUILD_DIR%"
mkdir "%RELEASE_DIR%\devign_infer"
mkdir "%RELEASE_DIR%\models"

echo.
echo [1/4] Copying scanner files...
xcopy /s /e /q "%ROOT_DIR%\devign_pipeline\devign_infer\*" "%RELEASE_DIR%\devign_infer\"
copy /y "%ROOT_DIR%\devign_pipeline\devign_scan.py" "%RELEASE_DIR%\"
copy /y "%ROOT_DIR%\requirements-inference.txt" "%RELEASE_DIR%\requirements.txt"

echo [2/4] Copying model files...
if exist "%ROOT_DIR%\models\best_model.pt" (
    copy /y "%ROOT_DIR%\models\best_model.pt" "%RELEASE_DIR%\models\"
    echo   [OK] best_model.pt
)
if exist "%ROOT_DIR%\models\vocab.json" (
    copy /y "%ROOT_DIR%\models\vocab.json" "%RELEASE_DIR%\models\"
    echo   [OK] vocab.json
)
if exist "%ROOT_DIR%\models\config.json" (
    copy /y "%ROOT_DIR%\models\config.json" "%RELEASE_DIR%\models\"
    echo   [OK] config.json
)

echo [3/4] Creating zip packages...
cd "%BUILD_DIR%"

REM Create code-only package
powershell -Command "Compress-Archive -Path 'release\devign_infer','release\devign_scan.py','release\requirements.txt' -DestinationPath 'devign-scanner-code.zip' -Force"
echo   [OK] devign-scanner-code.zip

REM Create full package
powershell -Command "Compress-Archive -Path 'release\*' -DestinationPath 'devign-scanner-full.zip' -Force"
echo   [OK] devign-scanner-full.zip

REM Create models-only package
if exist "release\models\best_model.pt" (
    powershell -Command "Compress-Archive -Path 'release\models' -DestinationPath 'devign-models.zip' -Force"
    echo   [OK] devign-models.zip
)

echo.
echo [4/4] Package sizes:
for %%f in ("%BUILD_DIR%\*.zip") do (
    echo   %%~nxf: %%~zf bytes
)

echo.
echo ==========================================
echo Build completed!
echo.
echo Output files in: %BUILD_DIR%
echo.
echo To publish to GitHub:
echo   gh release create %VERSION% build\*.zip
echo ==========================================

endlocal
