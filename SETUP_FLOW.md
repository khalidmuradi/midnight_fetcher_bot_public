# Setup Flow - With GPU Support

## 🚀 Quick Start

Just run:
```cmd
setup.cmd
```

The script will guide you through everything, including GPU setup!

---

## 📋 Setup Flow Diagram

```
START: setup.cmd
    ↓
[1/6] Check Node.js
    ├─ Found → Continue
    └─ Not Found → Install Node.js → Run setup.cmd again
    ↓
[2/6] Check hash-server.exe
    ├─ Found → Continue
    └─ Not Found → ERROR (should be in repo)
    ↓
[3/6] Check CUDA (GPU Support)
    ├─ CUDA Found →
    │   ├─ Build GPU now? (y/N)
    │   │   ├─ Yes → Run build-gpu.cmd → Continue
    │   │   └─ No → Skip (can build later) → Continue
    │   └─ Continue to step 4
    │
    └─ CUDA Not Found →
        ├─ Install CUDA now? (y/N)
        │   ├─ Yes →
        │   │   ├─ Open browser (download CUDA)
        │   │   ├─ User installs CUDA
        │   │   ├─ RESTART TERMINAL
        │   │   └─ Run setup.cmd again
        │   │
        │   └─ No → Continue CPU-only → Step 4
        └─ Continue to step 4
    ↓
[4/6] Install npm dependencies
    ↓
[5/6] Create directories (secure, storage, logs)
    ↓
[6/6] Start services (hash server + Next.js)
    ↓
DONE: Mining bot running!
```

---

## 🎮 User Experience Scenarios

### Scenario 1: No GPU, Don't Want GPU
```
[3/6] Checking for CUDA GPU support (optional)...
CUDA not detected - GPU mining will not be available

GPU mining can provide 100-200x faster hash rates!

To enable GPU mining, you need CUDA Toolkit 12.0+:
  - Download size: ~3 GB
  - Install time: 10-15 minutes
  - Requires: NVIDIA GPU (GTX 1000 series or newer)

Install CUDA Toolkit now? (y/N): N
```
**Result:** Continues with CPU-only setup ✅

---

### Scenario 2: No GPU, Want to Install CUDA
```
[3/6] Checking for CUDA GPU support (optional)...
CUDA not detected - GPU mining will not be available

Install CUDA Toolkit now? (y/N): y

Opening CUDA download page...

Installation Steps:
  1. Download CUDA Toolkit 12.0.5
  2. Run the installer (accept defaults)
  3. RESTART this terminal after installation
  4. Run setup.cmd again

After installing CUDA, please:
  1. Close this terminal window
  2. Open a NEW terminal window
  3. Run: setup.cmd

Press any key to continue...
```
**Result:** Downloads CUDA, waits for installation, then reruns setup ✅

---

### Scenario 3: CUDA Installed, Want GPU Build
```
[3/6] Checking for CUDA GPU support (optional)...
CUDA Toolkit detected!
Cuda compilation tools, release 12.0, V12.0.140

GPU mining is available! You can enable it with:
  1. Run: build-gpu.cmd
  2. Configure: lib\mining\config.ts (enableGpu: true)

Build with GPU support now? (y/N): y

Building HashEngine with GPU support...

========================================
 HashEngine GPU Build Script
========================================

[1/5] Checking CUDA installation...
CUDA found!

[2/5] Checking Rust installation...
Rust found!

[3/5] Building HashEngine with GPU support...
   Compiling cudarc v0.11.0
   Compiling HashEngine-napi v0.1.0
    Finished release [optimized] target(s) in 3m 42s

[4/5] Copying GPU-enabled library...
[OK] Library copied to index.node

[5/5] Testing GPU availability...
GPU Available: true

========================================
 Build Complete!
========================================

GPU build successful! GPU mining will be available.
```
**Result:** GPU mining ready! ✅

---

### Scenario 4: CUDA Installed, Skip GPU Build (Build Later)
```
[3/6] Checking for CUDA GPU support (optional)...
CUDA Toolkit detected!

Build with GPU support now? (y/N): N

Skipping GPU build. You can build later with: build-gpu.cmd
```
**Result:** CPU-only for now, can build GPU later ✅

---

## 🔄 Common Workflows

### First-Time User (No CUDA)
1. Run `setup.cmd`
2. Script detects no CUDA
3. Choose "y" to install CUDA
4. Browser opens → Download CUDA
5. Install CUDA (10-15 min)
6. **Close terminal, open new one**
7. Run `setup.cmd` again
8. Script detects CUDA
9. Choose "y" to build GPU
10. GPU mining ready! 🚀

**Time:** 15-20 minutes

---

### First-Time User (Already Has CUDA)
1. Run `setup.cmd`
2. Script detects CUDA
3. Choose "y" to build GPU
4. GPU mining ready! 🚀

**Time:** 5-10 minutes

---

### First-Time User (No GPU / Don't Care)
1. Run `setup.cmd`
2. Script detects no CUDA
3. Choose "N" (skip)
4. CPU mining ready!

**Time:** 2-3 minutes

---

### Add GPU Later
**Already ran setup.cmd with CPU-only:**

1. Install CUDA: https://developer.nvidia.com/cuda-downloads
2. Restart terminal
3. Run: `build-gpu.cmd`
4. GPU mining ready! 🚀

**Time:** 15-20 minutes (including CUDA install)

---

## 📊 Decision Tree

```
Do you have NVIDIA GPU?
├─ Yes → Do you want 100-200x faster mining?
│   ├─ Yes → Install CUDA → Build GPU → FAST! 🚀
│   └─ No → Skip GPU → CPU mining
│
└─ No → CPU mining only
```

---

## 🛠️ Troubleshooting

### Problem: CUDA Installed but Not Detected

**Symptoms:**
```
CUDA not detected - GPU mining will not be available
```

**Solution:**
```cmd
REM 1. Verify CUDA installed
dir "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"

REM 2. Check if nvcc is in PATH
where nvcc

REM 3. If not found, restart terminal (PATH update needed)
REM    Close terminal → Open new terminal → Try again

REM 4. If still not found, manually add to PATH:
set PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0\bin;%PATH%

REM 5. Run setup.cmd again
setup.cmd
```

---

### Problem: GPU Build Failed

**Symptoms:**
```
WARNING: GPU build failed. Continuing with CPU-only setup...
```

**Solution:**
```cmd
REM 1. Check CUDA version
nvcc --version

REM 2. Should show CUDA 12.0 or later
REM    If older, install CUDA 12.0.5+

REM 3. Check Visual Studio Build Tools installed
REM    Download: https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2022

REM 4. Try manual GPU build
build-gpu.cmd

REM 5. Check error messages for specific issues
```

---

### Problem: Setup Exits After CUDA Install Prompt

**This is intentional!**

After choosing to install CUDA (y), setup exits so you can:
1. Download and install CUDA
2. Restart terminal (PATH update)
3. Run `setup.cmd` again

**Just run setup.cmd again after CUDA installation.**

---

## 📖 Related Documentation

- **GPU_QUICKSTART.md** - 5-minute GPU setup guide
- **GPU_SETUP.md** - Complete GPU documentation
- **GPU_EXPLAINED.md** - Technical details about GPU mining
- **build-gpu.cmd** - Standalone GPU build script

---

## ✅ Summary

**The setup process is now fully automated:**

1. ✅ Detects CUDA automatically
2. ✅ Offers to install CUDA if missing
3. ✅ Offers to build GPU if CUDA present
4. ✅ Gracefully continues with CPU-only if user declines
5. ✅ Shows clear next steps at every stage
6. ✅ Can add GPU support later without reinstalling

**Just run `setup.cmd` and follow the prompts! 🚀**
