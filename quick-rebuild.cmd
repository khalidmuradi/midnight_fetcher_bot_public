@echo off
REM Quick rebuild without cargo clean (faster)

echo ========================================
echo  Quick GPU Rebuild
echo ========================================
echo.

where cl.exe >nul 2>&1
if errorlevel 1 (
    echo ERROR: Must run from VS Developer Command Prompt!
    echo.
    pause
    exit /b 1
)

echo Building...
cd hashengine
cargo build --release --features gpu
if %ERRORLEVEL% NEQ 0 (
    echo Build FAILED!
    cd ..
    pause
    exit /b 1
)
cd ..

echo.
echo Copying library...
copy /Y "hashengine\target\release\HashEngine_napi.dll" "index.node"

echo.
echo ========================================
echo  Testing...
echo ========================================
node test-gpu-validation.js

pause
