@echo off
REM Clean rebuild of GPU library
REM Run this from x64 Native Tools Command Prompt for VS 2022

echo ========================================
echo  Clean GPU Rebuild
echo ========================================
echo.

echo [1/4] Checking Visual Studio C++ compiler...
where cl.exe >nul 2>&1
if errorlevel 1 (
    echo ERROR: cl.exe not found!
    echo You must run this from "x64 Native Tools Command Prompt for VS 2022"
    pause
    exit /b 1
)
echo Visual Studio C++ compiler: OK
echo.

echo [2/4] Cleaning previous build...
cd hashengine
cargo clean
cd ..
echo Clean complete.
echo.

echo [3/4] Rebuilding with GPU support...
cd hashengine
cargo build --release --features gpu
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Build failed!
    cd ..
    pause
    exit /b 1
)
cd ..
echo Build successful!
echo.

echo [4/4] Copying library...
copy /Y "hashengine\target\release\HashEngine_napi.dll" "index.node"
echo.

echo ========================================
echo  Testing GPU Hash Correctness
echo ========================================
echo.
node test-gpu-validation.js

echo.
pause
