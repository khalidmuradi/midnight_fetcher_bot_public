# GPU Mining - Quick Start

Get GPU mining running in 5 minutes!

## ⚡ Quick Setup (Windows)

### 1. Install CUDA Toolkit
```cmd
REM Download and install CUDA 12.0.5+
https://developer.nvidia.com/cuda-downloads

REM Verify installation
nvcc --version
nvidia-smi
```

### 2. Build with GPU Support
```cmd
REM Run the automated build script
build-gpu.cmd
```

That's it! GPU mining will auto-enable when you start the bot.

### 3. Start Mining
```cmd
npm run dev
```

Watch for:
```
[HashEngineGPU] GPU: NVIDIA GeForce RTX 4090
[Orchestrator] ✓ GPU Mining Enabled
[Orchestrator] [GPU] Worker 0: Starting GPU mining
```

---

## 🎮 Configuration

Edit `lib/mining/config.ts`:

```typescript
enableGpu: true,        // Enable/disable GPU
gpuBatchSize: 100000,   // Hashes per GPU batch
gpuWorkers: 4,          // Number of GPU workers
```

**Recommended settings by GPU:**

| GPU Model | Batch Size | Workers | Expected H/s |
|-----------|------------|---------|--------------|
| RTX 3060  | 50,000     | 2-4     | 50K-100K     |
| RTX 3070  | 100,000    | 4-6     | 100K-200K    |
| RTX 4070  | 200,000    | 6-8     | 200K-400K    |
| RTX 4090  | 500,000    | 8-16    | 500K-1M      |

---

## 📊 Verify It's Working

### Check GPU Usage
```cmd
nvidia-smi

REM You should see:
REM - GPU Utilization: 80-100%
REM - Memory Usage: 2-8 GB
REM - Power: 200-400W
```

### Check Hash Rate
Look for console output:
```
[Orchestrator] [GPU] Worker 0: 1,000,000 hashes @ 185,000 H/s
```

Compare to CPU-only:
```
[Orchestrator] Worker 0: 50,000 hashes @ 3,500 H/s
```

**GPU should be 10-200x faster!**

---

## 🐛 Troubleshooting

### GPU Not Detected
```
[Orchestrator] ⚠️  No CUDA GPU detected
```

**Fix:**
1. Install CUDA: https://developer.nvidia.com/cuda-downloads
2. Rebuild: `build-gpu.cmd`
3. Restart terminal

### Build Errors
```
error: CUDA not found
```

**Fix:**
```cmd
set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0
build-gpu.cmd
```

### Out of Memory
```
CUDA out of memory
```

**Fix:** Reduce batch size in `lib/mining/config.ts`:
```typescript
gpuBatchSize: 10000,  // Start small
```

---

## 📖 Full Documentation

See [GPU_SETUP.md](./GPU_SETUP.md) for:
- Detailed installation steps
- Performance tuning
- Advanced configuration
- Troubleshooting guide
- Architecture details

---

## 💡 Tips

- **Run both CPU + GPU** for maximum performance
- **Monitor temperature**: Keep GPU <85°C
- **Start with default settings**, tune later
- **GPU mines different nonce range** than CPU (no overlap)
- **First solution wins** - CPU or GPU

---

**Expected Performance:**
- CPU Only: ~3,000-5,000 H/s
- GPU (RTX 4090): ~500,000-1,000,000 H/s
- **100-200x faster! 🚀**
