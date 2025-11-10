# GPU Build Quick Reference

## TL;DR - Just Want It To Work?

```cmd
1. Search for "x64 Native Tools Command Prompt for VS 2022" in Windows Start Menu
2. Open it
3. cd C:\Users\paddy\OneDrive\Documents\Repos\midnight_fetcher_bot
4. build-gpu.cmd
5. Done!
```

## Why Do I Need Visual Studio Command Prompt?

CUDA's `nvcc` compiler needs Microsoft's C++ compiler (`cl.exe`) to work. The Visual Studio Developer Command Prompt automatically adds it to PATH.

## What If I Don't Have Visual Studio?

Install Visual Studio Build Tools (free):
1. Go to https://visualstudio.microsoft.com/downloads/
2. Download "Build Tools for Visual Studio 2022"
3. Run installer
4. Select "Desktop development with C++"
5. Install (~7 GB)
6. Restart terminal
7. Run `build-gpu.cmd`

## Verification Checklist

Before running `build-gpu.cmd`, verify:

```cmd
# Check CUDA (should show version)
nvcc --version

# Check Visual Studio C++ compiler (should show path)
where cl.exe

# Check Rust (should show version)
cargo --version
```

If all three work, you're ready to build!

## What Gets Built?

1. **CUDA Compilation**: `hashengine_gpu.cu` → `hashengine_gpu.ptx`
   - This happens during `cargo build`
   - PTX is CUDA intermediate code (like LLVM IR)
   - Embedded into Rust binary

2. **Rust Compilation**: `gpu.rs` + PTX → `HashEngine_napi.dll`
   - Rust loads PTX at runtime
   - JIT compiles to native GPU code
   - Creates NAPI bindings for Node.js

3. **Final Library**: `index.node`
   - Node.js native module
   - Contains GPU kernels + Rust code
   - Ready to use from TypeScript

## Build Output Explained

```
cargo:warning=Compiling CUDA kernels to PTX...
  └─ nvcc is compiling your CUDA code

cargo:warning=✓ CUDA kernels compiled successfully
  └─ PTX file created successfully

Compiling HashEngine-napi v0.1.0
  └─ Rust is compiling and embedding PTX

[OK] Build successful!
  └─ hashengine/target/release/HashEngine_napi.dll created

[OK] Library copied to index.node
  └─ Ready to use from Node.js
```

## Common Issues

### "Cannot find compiler 'cl.exe'"
**Problem**: Not using VS Developer Command Prompt
**Solution**: Use "x64 Native Tools Command Prompt for VS"

### "nvcc: command not found"
**Problem**: CUDA not installed or not in PATH
**Solution**: Install CUDA Toolkit, restart terminal

### "error: linker 'link.exe' not found"
**Problem**: Missing Visual Studio linker
**Solution**: Install Visual Studio Build Tools with C++ support

## Advanced: Adding to System PATH

If you want `cl.exe` in your normal terminal:

1. Find Visual Studio installation (e.g., `C:\Program Files\Microsoft Visual Studio\2022\BuildTools\`)
2. Run this in normal terminal (adjust path):
```cmd
"C:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
```
3. Now `cl.exe` is available
4. Run `build-gpu.cmd`

**Note**: This only lasts for the current terminal session.

## Files to Check

After successful build, verify these files exist:

```
✓ hashengine/target/release/HashEngine_napi.dll  (Rust library)
✓ index.node  (Node.js module, copy of above)
✓ hashengine/target/release/build/.../hashengine_gpu.ptx  (CUDA kernel)
```

## Testing

After build:
```cmd
# Quick test
node -e "console.log(require('./index.node').gpuAvailable())"

# Full test
test-gpu.cmd
```

Expected output: `true` (first command) or detailed test results (second command)

## Rebuilding

Clean build:
```cmd
cd hashengine
cargo clean
cd ..
build-gpu.cmd
```

Quick rebuild (after code changes):
```cmd
build-gpu.cmd
```

## Integration with setup.cmd

The `setup.cmd` script will offer to run `build-gpu.cmd` if CUDA is detected:

```
[3/6] Checking for CUDA GPU support (optional)...
CUDA Toolkit detected!
...
Build with GPU support now? (y/N): y
```

If you say yes, it automatically runs the GPU build process.

## Performance Check

After building, check performance:

```javascript
// CPU (baseline)
const start1 = Date.now();
const hash1 = hashEngine.hashPreimage("test");
console.log(`CPU: ${Date.now() - start1}ms`);

// GPU (batch)
const start2 = Date.now();
const hashes = hashEngine.hashBatchGpu(Array(1000).fill("test"));
console.log(`GPU (1000): ${Date.now() - start2}ms`);
console.log(`Per hash: ${(Date.now() - start2) / 1000}ms`);
```

Expected GPU to be 100-200x faster per hash in large batches.

## Summary

**Minimum Requirements**:
1. NVIDIA GPU (GTX 1000+)
2. CUDA Toolkit 12.0+
3. Visual Studio Build Tools (C++)
4. Rust (already installed)

**Build Command**:
```cmd
build-gpu.cmd  (from VS Developer Command Prompt)
```

**Verification**:
```cmd
test-gpu.cmd
```

That's it! 🚀
