# HashEngineGPU Setup Guide

Complete guide to setting up GPU-accelerated mining for your midnight fetcher bot.

## 🎯 Overview

The HashEngineGPU system enables **dual CPU+GPU mining**, allowing your NVIDIA GPU to mine alongside your CPU for maximum hash rate. The system automatically:
- Detects GPU availability
- Coordinates nonce ranges between CPU and GPU (no overlap)
- Manages separate worker pools for each
- Accepts solutions from whichever finds it first

### Expected Performance
- **CPU Only**: ~3,000-5,000 H/s (11 workers)
- **GPU (RTX 3060)**: ~50,000-200,000 H/s (**10-40x faster**)
- **GPU (RTX 4090)**: ~500,000-1,000,000 H/s (**100-200x faster**)
- **Both Combined**: Maximum utilization of all hardware

---

## 📋 Prerequisites

### 1. NVIDIA GPU Requirements
- **GPU**: NVIDIA GPU with CUDA support (Compute Capability 6.0+)
  - Recommended: RTX 3060 or better
  - Check your GPU: https://developer.nvidia.com/cuda-gpus
- **VRAM**: Minimum 4GB (8GB+ recommended)

### 2. CUDA Toolkit
You need CUDA 12.0.5 or later installed.

**Windows Installation:**
1. Download CUDA Toolkit: https://developer.nvidia.com/cuda-downloads
2. Select: Windows -> x86_64 -> Your Windows Version -> exe (local)
3. Run installer (will take 10-15 minutes)
4. Verify installation:
   ```cmd
   nvcc --version
   ```
   Should show: `cuda_12.0` or later

**Verify CUDA is working:**
```cmd
nvidia-smi
```
Should display your GPU info and CUDA version.

### 3. Rust CUDA Support
The Rust compiler needs to find CUDA libraries.

**Windows:**
```cmd
set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0
set PATH=%CUDA_PATH%\bin;%PATH%
```

Add these permanently via System Environment Variables or add to your shell profile.

---

## 🔧 Building with GPU Support

### Step 1: Build Rust HashEngine with GPU Feature

```cmd
cd hashengine
cargo build --release --features gpu
```

This will:
- Compile CUDA kernels (Blake2b, Argon2, VM)
- Link against CUDA runtime
- Create `HashEngine-napi` library with GPU functions

**Expected output:**
```
Compiling cudarc v0.11.0
Compiling HashEngine-napi v0.1.0
Finished release [optimized] target(s) in 5m 23s
```

### Step 2: Copy GPU-enabled Library

```cmd
copy /Y "hashengine\target\release\HashEngine_napi.dll" "index.node"
```

**Verify GPU functions are available:**
```cmd
node -e "const engine = require('./index.node'); console.log('GPU Available:', engine.gpuAvailable())"
```

Should print: `GPU Available: true`

---

## ⚙️ Configuration

### Mining Configuration
Edit `lib/mining/config.ts` to customize GPU settings:

```typescript
export const DEFAULT_MINING_CONFIG: MiningConfig = {
  // CPU settings
  cpuWorkers: 11,
  cpuBatchSize: 300,

  // GPU settings
  enableGpu: true,              // Set to false to disable GPU
  gpuBatchSize: 100000,         // Increase for faster GPUs (RTX 4090: 500000)
  gpuWorkers: 4,                // Number of parallel GPU batches

  // Nonce ranges (prevent CPU/GPU overlap)
  cpuNonceStart: 0n,
  cpuNonceEnd: 10_000_000_000n, // CPU mines 0 to 10 billion

  gpuNonceStart: 10_000_000_000n,
  gpuNonceEnd: 1_000_000_000_000n, // GPU mines 10B to 1T
};
```

**Tuning Tips:**
- **RTX 3060**: `gpuBatchSize: 50000`, `gpuWorkers: 2-4`
- **RTX 4070/4080**: `gpuBatchSize: 200000`, `gpuWorkers: 4-8`
- **RTX 4090**: `gpuBatchSize: 500000`, `gpuWorkers: 8-16`

Monitor GPU memory usage with `nvidia-smi` and adjust if you see OOM errors.

---

## 🚀 Running with GPU

### Start Mining
Just start mining normally - GPU will auto-initialize if available:

```cmd
npm run dev
```

**Console output (successful GPU init):**
```
[Orchestrator] ROM ready
[Orchestrator] ========================================
[Orchestrator] INITIALIZING GPU MINING
[Orchestrator] ========================================
[HashEngineGPU] Initializing CUDA device...
[HashEngineGPU] GPU: NVIDIA GeForce RTX 4090
[HashEngineGPU] Compute capability: 8.9
[HashEngineGPU] Memory: 24564 MB
[HashEngineGPU] Compiling CUDA kernels...
[HashEngineGPU] Kernels compiled successfully
[HashEngineGPU] Uploading ROM to GPU (65536 bytes)...
[HashEngineGPU] ROM uploaded successfully
[HashEngineGPU] Batch size: 100000
[HashEngineGPU] Initialization complete!
[Orchestrator] ✓ GPU Mining Enabled
[Orchestrator]   Device: NVIDIA GeForce RTX 4090
[Orchestrator]   Workers: 4
[Orchestrator]   Batch Size: 100,000
[Orchestrator] ========================================
```

### Monitor Mining
Watch for GPU worker output:
```
[Orchestrator] [GPU] Worker 0 for Address 5: Starting GPU mining
[Orchestrator] [GPU] Worker 0: Nonce range 10,000,000,000 to 257,500,000,000
[Orchestrator] [GPU] Worker 0: 1,000,000 hashes @ 185,000 H/s
```

---

## 🐛 Troubleshooting

### GPU Not Detected
**Symptom:** `[Orchestrator] ⚠️  No CUDA GPU detected - GPU mining disabled`

**Solutions:**
1. Verify CUDA installed: `nvcc --version`
2. Check GPU visible: `nvidia-smi`
3. Rebuild with GPU feature: `cargo build --release --features gpu`
4. Ensure CUDA_PATH environment variable is set

### Compilation Errors
**Error:** `error: linker 'link.exe' not found`

**Solution:** Install Visual Studio Build Tools
```cmd
https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2022
```
Select "Desktop development with C++"

**Error:** `CUDA not found`

**Solution:** Set CUDA_PATH:
```cmd
set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0
```

### Out of Memory (OOM)
**Error:** `CUDA out of memory`

**Solution:** Reduce `gpuBatchSize` in config:
```typescript
gpuBatchSize: 10000,  // Start small, increase gradually
```

### Low GPU Utilization
**Symptom:** `nvidia-smi` shows GPU usage <50%

**Solutions:**
- Increase `gpuWorkers` (more parallel batches)
- Increase `gpuBatchSize` (more hashes per batch)
- Check if CPU is bottleneck (preimage generation)

---

## 📊 Benchmarking

### Test GPU Performance
```typescript
// In browser console or Node REPL
const { hashEngineGPU } = require('@/lib/hash/engine-gpu');

// Initialize
await hashEngineGPU.init(10000);

// Test batch
const preimages = Array(10000).fill('test_preimage');
const start = Date.now();
await hashEngineGPU.hashBatch(preimages);
const elapsed = Date.now() - start;
console.log(`Hash rate: ${Math.round(10000 / elapsed * 1000)} H/s`);
```

### Compare CPU vs GPU
Run mining for 5 minutes each:
1. Disable GPU: `enableGpu: false` in config
2. Run and note hash rate
3. Enable GPU: `enableGpu: true`
4. Run and note combined hash rate

Expected speedup: **10-200x** depending on GPU model.

---

## 🔄 Updating

When updating the codebase:

1. Pull latest changes
2. Rebuild Rust with GPU:
   ```cmd
   cd hashengine
   cargo build --release --features gpu
   copy /Y "target\release\HashEngine_napi.dll" "..\index.node"
   ```
3. Restart mining

---

## 💡 Tips & Best Practices

### Optimal Configuration
- **Always run both CPU + GPU** for maximum hash rate
- **Monitor temperatures**: Keep GPU <85°C (use MSI Afterburner)
- **Power limit**: Consider reducing power limit to 80-90% for better efficiency
- **Nonce ranges**: CPU and GPU automatically use different ranges (no conflicts)

### Production Setup
```typescript
// High-end setup (RTX 4090)
{
  cpuWorkers: 16,           // All CPU cores
  cpuBatchSize: 500,
  enableGpu: true,
  gpuBatchSize: 500000,     // Large batches for high-end GPU
  gpuWorkers: 16,           // Many parallel batches
}

// Mid-range setup (RTX 3060)
{
  cpuWorkers: 8,
  cpuBatchSize: 300,
  enableGpu: true,
  gpuBatchSize: 50000,
  gpuWorkers: 4,
}

// Low-power setup (Laptop)
{
  cpuWorkers: 4,
  cpuBatchSize: 200,
  enableGpu: true,
  gpuBatchSize: 20000,
  gpuWorkers: 2,
}
```

---

## 📚 Architecture

### How It Works
1. **ROM Initialization**: ROM uploaded to GPU memory once per challenge
2. **Worker Spawning**: CPU workers (0-10) and GPU workers (1000-1003) spawn
3. **Nonce Allocation**:
   - CPU: 0 to 10 billion
   - GPU: 10 billion to 1 trillion
4. **Parallel Mining**: Both mine simultaneously, different ranges
5. **Solution Detection**: First to find solution wins, others stop
6. **Submission**: Same submission logic for both CPU and GPU

### Files
- `hashengine/src/gpu.rs` - Rust GPU wrapper
- `hashengine/kernels/*.cu` - CUDA kernels
- `lib/hash/engine-gpu.ts` - TypeScript GPU interface
- `lib/mining/config.ts` - Mining configuration
- `lib/mining/orchestrator.ts` - Dual-engine coordination

---

## ❓ FAQ

**Q: Can I use AMD GPU?**
A: Not currently. Only NVIDIA CUDA is supported. AMD ROCm support could be added in the future.

**Q: Will this work on Mac?**
A: No. CUDA is NVIDIA-only and doesn't work on macOS (even with eGPU). Use CPU mining.

**Q: Does GPU mining use more power?**
A: Yes, significantly (200-450W vs 65-150W CPU). But hash rate increase is worth it.

**Q: Can I mine on multiple GPUs?**
A: Not yet, but easy to add. File a GitHub issue if you need this.

**Q: Is GPU mining more profitable?**
A: **YES**. 100x more hashes = 100x more solutions = 100x more rewards.

---

## 🆘 Support

If you encounter issues:

1. Check this document first
2. Enable debug logging: `DEBUG=* npm run dev`
3. Check GPU status: `nvidia-smi`
4. Create GitHub issue with:
   - GPU model
   - CUDA version (`nvcc --version`)
   - Error messages
   - Console output

---

**Happy GPU Mining! 🚀**
