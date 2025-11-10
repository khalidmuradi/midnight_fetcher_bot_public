@echo off
REM Auto-build GPU with Visual Studio environment
REM This script automatically sets up the VS environment and builds

echo Setting up Visual Studio environment...
call "C:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"

if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Could not set up Visual Studio environment
    echo Please run from x64 Native Tools Command Prompt instead
    exit /b 1
)

echo.
echo ========================================
echo  Building HashEngine with GPU Support
echo ========================================
echo.

cd hashengine
cargo build --release --features gpu

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Build failed!
    cd ..
    exit /b 1
)

cd ..

REM Copy DLL
if exist "hashengine\target\release\HashEngine_napi.dll" (
    copy /Y "hashengine\target\release\HashEngine_napi.dll" "index.node"
    echo.
    echo [OK] Build successful! Library copied to index.node
    echo.
) else (
    echo [ERROR] Build output not found!
    exit /b 1
)

REM Test GPU
echo Testing GPU availability...
node -e "try { const engine = require('./index.node'); console.log('GPU Available:', engine.gpuAvailable()); } catch(e) { console.error('Error:', e.message); }"
echo.
echo Build complete!
