@echo off
REM Clean rebuild and test GPU with reference Blake2b implementation

echo ========================================
echo  GPU Rebuild with Reference Blake2b
echo ========================================
echo.

echo [1/5] Checking Visual Studio C++ compiler...
where cl.exe >nul 2>&1
if errorlevel 1 (
    echo ERROR: Must run from "x64 Native Tools Command Prompt for VS 2022"
    echo.
    echo Please:
    echo   1. Search for "x64 Native Tools Command Prompt for VS" in Start Menu
    echo   2. Open it
    echo   3. cd C:\Users\paddy\OneDrive\Documents\Repos\midnight_fetcher_bot
    echo   4. test-and-rebuild.cmd
    echo.
    pause
    exit /b 1
)
echo Visual Studio C++ found!
echo.

echo [2/5] Cleaning previous build...
cd hashengine
cargo clean
cd ..
echo.

echo [3/5] Building with reference Blake2b implementation...
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
echo.

echo [4/5] Copying library...
copy /Y "hashengine\target\release\HashEngine_napi.dll" "index.node"
echo.

echo [5/5] Testing GPU vs CPU hash correctness...
node test-gpu-validation.js

echo.
echo ========================================
echo  Test Complete!
echo ========================================
echo.
pause
