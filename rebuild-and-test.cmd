@echo off
REM Build and test GPU implementation
REM Must be run from VS Developer Command Prompt

echo ========================================
echo  GPU Rebuild and Test
echo ========================================
echo.

where cl.exe >nul 2>&1
if errorlevel 1 (
    echo ERROR: Must run from VS Developer Command Prompt!
    echo.
    echo Please run from: x64 Native Tools Command Prompt for VS 2022
    pause
    exit /b 1
)

echo [1/3] Building GPU library...
cd hashengine
cargo build --release --features gpu
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo BUILD FAILED!
    cd ..
    pause
    exit /b 1
)
cd ..

echo.
echo [2/3] Copying library to index.node...
copy /Y "hashengine\target\release\HashEngine_napi.dll" "index.node"

echo.
echo [3/3] Testing GPU vs CPU...
echo ========================================
node test-blake2b-basic.js

echo.
echo ========================================
echo Done!
pause
