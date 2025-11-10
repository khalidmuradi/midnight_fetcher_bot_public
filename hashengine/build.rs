// Build script for CUDA kernels and NAPI bindings
use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    // Compile CUDA kernels if GPU feature is enabled
    #[cfg(feature = "gpu")]
    compile_cuda();

    // NAPI bindings
    #[cfg(feature = "napi-bindings")]
    {
        println!("cargo:warning=NAPI bindings require napi-build dependency");
    }
}

#[cfg(feature = "gpu")]
fn compile_cuda() {
    println!("cargo:rerun-if-changed=kernels/hashengine_gpu.cu");

    // Check if nvcc is available
    let nvcc_available = Command::new("nvcc")
        .arg("--version")
        .output()
        .is_ok();

    if !nvcc_available {
        println!("cargo:warning=CUDA compiler (nvcc) not found. GPU feature will not work.");
        println!("cargo:warning=Install CUDA Toolkit from https://developer.nvidia.com/cuda-downloads");
        return;
    }

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let cuda_source = "kernels/hashengine_gpu.cu";
    let ptx_output = out_dir.join("hashengine_gpu.ptx");

    println!("cargo:warning=Compiling CUDA kernels to PTX...");
    println!("cargo:warning=NOTE: This requires Visual Studio C++ compiler (cl.exe) in PATH");
    println!("cargo:warning=Run this from 'x64 Native Tools Command Prompt for VS' if you see cl.exe errors");

    // Compile CUDA to PTX (intermediate representation)
    let output = Command::new("nvcc")
        .arg(cuda_source)
        .arg("--ptx")
        .arg("-o")
        .arg(&ptx_output)
        .arg("--gpu-architecture=sm_60") // Minimum compute capability 6.0 (GTX 1000 series+)
        .arg("--use_fast_math")
        .arg("-O3")
        .output();

    match output {
        Ok(output) if output.status.success() => {
            println!("cargo:warning=✓ CUDA kernels compiled successfully");
            println!("cargo:rustc-env=CUDA_PTX_PATH={}", ptx_output.display());
        }
        Ok(output) => {
            let stderr = String::from_utf8_lossy(&output.stderr);
            let stdout = String::from_utf8_lossy(&output.stdout);
            println!("cargo:warning=CUDA compilation failed!");
            println!("cargo:warning=stdout: {}", stdout);
            println!("cargo:warning=stderr: {}", stderr);

            if stderr.contains("cl.exe") || stdout.contains("cl.exe") {
                println!("cargo:warning=");
                println!("cargo:warning=ERROR: Visual Studio C++ compiler not found!");
                println!("cargo:warning=");
                println!("cargo:warning=To fix this:");
                println!("cargo:warning=1. Close this terminal");
                println!("cargo:warning=2. Open 'x64 Native Tools Command Prompt for VS 2022' from Start menu");
                println!("cargo:warning=3. Run your build command again from that terminal");
                println!("cargo:warning=");
                println!("cargo:warning=Or install Visual Studio Build Tools from:");
                println!("cargo:warning=https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2022");
            }

            panic!("Failed to compile CUDA kernels");
        }
        Err(e) => {
            println!("cargo:warning=Failed to run nvcc: {}", e);
            panic!("Failed to compile CUDA kernels");
        }
    }
}
