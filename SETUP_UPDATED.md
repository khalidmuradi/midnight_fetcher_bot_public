# Setup Script - GPU Integration Update

## ✅ What Was Updated

The `setup.cmd` script has been enhanced to support GPU mining setup:

### New Features Added

1. **CUDA Detection** (Step 3/6)
   - Automatically detects if CUDA Toolkit is installed
   - Displays CUDA version if found
   - Shows helpful messages if CUDA not detected

2. **Interactive GPU Build**
   - Asks user if they want to build with GPU support
   - Calls `build-gpu.cmd` if user chooses "y"
   - Gracefully continues with CPU-only if GPU build fails
   - Can be skipped and built later

3. **GPU Status Display**
   - Shows mining status at startup
   - Indicates if GPU mining is available
   - Shows whether GPU library is built
   - Always shows CPU mining as enabled

4. **Updated Documentation Links**
   - References `GPU_QUICKSTART.md` for GPU setup
   - Provides clear next steps for GPU enablement

---

## 📋 Setup Flow

### Previous Flow (CPU Only)
```
1. Check Node.js
2. Verify hash-server.exe
3. Install dependencies
4. Create directories
5. Start services
```

### New Flow (CPU + Optional GPU)
```
1. Check Node.js
2. Verify hash-server.exe
3. Check CUDA (NEW!)
   ├─ CUDA Found?
   │  ├─ Yes → Ask to build GPU
   │  │  ├─ User says Yes → Run build-gpu.cmd
   │  │  └─ User says No → Skip, can build later
   │  └─ No → Show CUDA install instructions
4. Install dependencies
5. Create directories
6. Start services (with GPU status display)
```

---

## 🎮 User Experience

### With CUDA Installed
```
[3/6] Checking for CUDA GPU support (optional)...
CUDA Toolkit detected!
Cuda compilation tools, release 12.0, V12.0.140

GPU mining is available! You can enable it with:
  1. Run: build-gpu.cmd
  2. Configure: lib\mining\config.ts (enableGpu: true)

Build with GPU support now? (y/N): _
```

**If user chooses 'y':**
```
Building HashEngine with GPU support...
[HashEngineGPU] Compiling CUDA kernels...
GPU build successful! GPU mining will be available.
```

**If user chooses 'n':**
```
Skipping GPU build. You can build later with: build-gpu.cmd
```

### Without CUDA
```
[3/6] Checking for CUDA GPU support (optional)...
CUDA not detected - GPU mining will not be available

To enable GPU mining:
  1. Install CUDA Toolkit 12.0+ from:
     https://developer.nvidia.com/cuda-downloads
  2. Run: build-gpu.cmd
  3. See GPU_QUICKSTART.md for details
```

### At Startup
```
================================================================================
                    Midnight Fetcher Bot - Ready!
================================================================================

Hash Service: http://127.0.0.1:9001/health
Web Interface: http://localhost:3000

Mining Status:
  - GPU Mining: Available (CUDA detected, library built)
  - CPU Mining: Enabled

The application will open in your default browser.
Press Ctrl+C to stop the Next.js server (hash server will continue running)

To stop hash server: taskkill /F /IM hash-server.exe
GPU Setup Guide: GPU_QUICKSTART.md
================================================================================
```

---

## 🔍 Status Messages

The script now shows one of three GPU status messages:

### 1. GPU Ready
```
- GPU Mining: Available (CUDA detected, library built)
```
Means: CUDA installed AND `index.node` exists (GPU-enabled library built)

### 2. GPU Available (Not Built)
```
- GPU Mining: Available (CUDA detected, run build-gpu.cmd)
```
Means: CUDA installed but `index.node` not built yet - needs `build-gpu.cmd`

### 3. GPU Not Available
```
- GPU Mining: Not available (CUDA not installed)
```
Means: CUDA Toolkit not detected - need to install CUDA first

---

## 🚀 Quick Commands

### First-Time Setup
```cmd
setup.cmd
```
Will detect GPU and offer to build with GPU support

### Add GPU Later
```cmd
build-gpu.cmd
```
Build GPU support after initial setup

### Check GPU Status
```cmd
node -e "const e = require('./index.node'); console.log('GPU:', e.gpuAvailable())"
```

---

## 📖 Documentation References

The setup script now references:
- `GPU_QUICKSTART.md` - Quick 5-minute GPU setup
- `GPU_SETUP.md` - Complete GPU documentation
- `build-gpu.cmd` - Automated GPU build

---

## ✨ Benefits

1. **Seamless Integration**: GPU setup is part of main setup flow
2. **Non-Intrusive**: Completely optional, doesn't break CPU-only setup
3. **User-Friendly**: Clear prompts and status messages
4. **Flexible**: Can skip during setup and add GPU later
5. **Informative**: Always shows current GPU/CPU mining status

---

## 🧪 Testing

To test the updated setup:

1. **Without CUDA** (should skip GPU gracefully):
   ```cmd
   setup.cmd
   ```
   Should show "CUDA not detected" and continue

2. **With CUDA** (should offer GPU build):
   ```cmd
   setup.cmd
   ```
   Should detect CUDA and ask to build

3. **GPU Already Built** (should show "available"):
   ```cmd
   setup.cmd
   ```
   Should show "GPU Mining: Available (library built)"

---

## 📝 Summary

**Updated File:** `setup.cmd`

**Changes:**
- ✅ Added CUDA detection (step 3/6)
- ✅ Interactive GPU build prompt
- ✅ GPU status in startup message
- ✅ Updated step numbering (now 6 steps instead of 5)
- ✅ Added documentation references
- ✅ Graceful fallback if GPU build fails

**Backward Compatible:** Yes - works exactly the same on systems without CUDA

**User Impact:** Positive - GPU setup is now integrated and easier
