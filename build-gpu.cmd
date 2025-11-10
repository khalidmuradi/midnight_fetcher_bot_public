@echo off
REM Build script for HashEngine with GPU support
REM Requires: CUDA Toolkit 12.0+ and Visual Studio Build Tools

echo ========================================
echo  HashEngine GPU Build Script
echo ========================================
echo.

REM Check if CUDA is installed (skip if called from setup.cmd)
echo [1/5] Checking CUDA installation...
if "%SKIP_CUDA_CHECK%"=="1" (
    echo CUDA already verified by setup.cmd
    echo.
) else (
    where nvcc >nul 2>&1
    if %ERRORLEVEL% NEQ 0 (
        echo [ERROR] CUDA not found! Please install CUDA Toolkit 12.0+
        echo Download from: https://developer.nvidia.com/cuda-downloads
        echo.
        echo After installation:
        echo   1. Restart this terminal
        echo   2. Run: build-gpu.cmd
        echo.
        pause
        exit /b 1
    )
    nvcc --version
    echo.
)

REM Check if Rust is installed
echo [2/5] Checking Rust installation...
where cargo >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Rust not found! Please install Rust from https://rustup.rs
    pause
    exit /b 1
)

cargo --version
echo.

REM Check for Visual Studio C++ compiler
echo [3/6] Checking for Visual Studio C++ compiler...
where cl.exe >nul 2>&1
if errorlevel 1 goto no_compiler
echo Visual Studio C++ compiler found!
echo.
goto compiler_ok

:no_compiler
echo.
echo [WARNING] Visual Studio C++ compiler (cl.exe) not in PATH!
echo.
echo CUDA requires the Visual Studio C++ compiler to build kernels.
echo.
echo To fix this, you have two options:
echo.
echo OPTION 1 (Recommended): Use Visual Studio Developer Command Prompt
echo   1. Close this window
echo   2. Search for "x64 Native Tools Command Prompt for VS" in Start Menu
echo   3. Run: cd /d "%CD%"
echo   4. Run: build-gpu.cmd
echo.
echo OPTION 2: Install Visual Studio Build Tools
echo   1. Download from: https://visualstudio.microsoft.com/downloads/
echo   2. Install "Desktop development with C++"
echo   3. Restart terminal and try again
echo.
pause
exit /b 1

:compiler_ok

REM Build HashEngine with GPU feature
echo [4/6] Building HashEngine with GPU support...
cd hashengine
cargo build --release --features gpu
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Build failed! Check error messages above.
    cd ..
    pause
    exit /b 1
)
cd ..
echo [OK] Build successful!
echo.

REM Copy DLL to project root
echo [5/6] Copying GPU-enabled library...
if exist "hashengine\target\release\HashEngine_napi.dll" (
    copy /Y "hashengine\target\release\HashEngine_napi.dll" "index.node"
    echo [OK] Library copied to index.node
) else (
    echo [ERROR] Build output not found!
    pause
    exit /b 1
)
echo.

REM Test GPU availability
echo [6/6] Testing GPU availability...
node -e "try { const engine = require('./index.node'); console.log('GPU Available:', engine.gpuAvailable()); } catch(e) { console.error('Error:', e.message); }"
echo.

echo ========================================
echo  Build Complete!
echo ========================================
echo.
echo GPU-accelerated HashEngine is ready!
echo Start mining with: npm run dev
echo.
echo GPU will auto-initialize if CUDA GPU detected.
echo See GPU_SETUP.md for configuration and troubleshooting.
echo.
pause
