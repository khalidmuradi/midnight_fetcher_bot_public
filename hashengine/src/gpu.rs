// HashEngineGPU - CUDA-accelerated mining engine
// Full implementation using compiled CUDA kernels

use crate::rom::Rom;
use std::sync::Arc;

#[cfg(feature = "gpu")]
use cudarc::driver::{CudaDevice, CudaSlice, LaunchAsync, LaunchConfig, CudaFunction};
#[cfg(feature = "gpu")]
use cudarc::nvrtc::Ptx;

/// GPU mining engine state
pub struct HashEngineGPU {
    #[cfg(feature = "gpu")]
    device: Arc<CudaDevice>,
    #[cfg(feature = "gpu")]
    kernel_func: CudaFunction,
    #[cfg(feature = "gpu")]
    rom_data_gpu: CudaSlice<u8>,
    #[cfg(feature = "gpu")]
    rom_digest_gpu: CudaSlice<u8>,
    #[cfg(feature = "gpu")]
    rom_size: u32,
    #[cfg(feature = "gpu")]
    nb_loops: u32,
    #[cfg(feature = "gpu")]
    nb_instrs: u32,
    batch_size: usize,
}

/// GPU device information
#[derive(Debug, Clone)]
pub struct GPUInfo {
    pub name: String,
    pub compute_capability: String,
    pub total_memory: usize,
    pub free_memory: usize,
}

impl HashEngineGPU {
    /// Check if GPU mining is available
    pub fn is_available() -> bool {
        #[cfg(feature = "gpu")]
        {
            use cudarc::driver::CudaDevice;
            CudaDevice::new(0).is_ok()
        }
        #[cfg(not(feature = "gpu"))]
        {
            false
        }
    }

    /// Initialize GPU mining engine with ROM
    #[cfg(feature = "gpu")]
    pub fn new(rom: &Rom, batch_size: usize) -> Result<Self, Box<dyn std::error::Error>> {
        println!("[HashEngineGPU] Initializing CUDA device...");

        // Initialize CUDA device (GPU 0)
        let device = CudaDevice::new(0)?;
        println!("[HashEngineGPU] Device: {}", device.name()?);

        // Load PTX compiled during build
        let ptx_str = include_str!(env!("CUDA_PTX_PATH"));
        let ptx = Ptx::from_src(ptx_str);

        device.load_ptx(ptx.clone(), "hashengine_gpu", &["hashengine_mine_batch"])?;
        let kernel_func = device.get_func("hashengine_gpu", "hashengine_mine_batch").unwrap();

        println!("[HashEngineGPU] CUDA kernels loaded successfully");

        // Copy ROM data to GPU
        let rom_data = rom.data();
        let rom_data_gpu = device.htod_copy(rom_data.to_vec())?;

        // Copy ROM digest to GPU
        let rom_digest = rom.digest.0;
        let rom_digest_gpu = device.htod_copy(rom_digest.to_vec())?;

        println!("[HashEngineGPU] ROM data uploaded to GPU ({} bytes)", rom_data.len());
        println!("[HashEngineGPU] Batch size: {}", batch_size);
        println!("[HashEngineGPU] GPU mining ready!");

        Ok(Self {
            device,
            kernel_func,
            rom_data_gpu,
            rom_digest_gpu,
            rom_size: rom_data.len() as u32,
            nb_loops: 8,  // Default from ROM params
            nb_instrs: 256,  // Default from ROM params
            batch_size,
        })
    }

    #[cfg(not(feature = "gpu"))]
    pub fn new(_rom: &Rom, _batch_size: usize) -> Result<Self, Box<dyn std::error::Error>> {
        Err("GPU support not compiled. Rebuild with --features gpu".into())
    }

    /// Hash a batch of preimages using GPU
    #[cfg(feature = "gpu")]
    pub fn hash_batch(&self, preimages: &[String]) -> Result<Vec<String>, Box<dyn std::error::Error>> {
        if preimages.is_empty() {
            return Ok(Vec::new());
        }

        let actual_batch_size = preimages.len().min(self.batch_size);

        // Prepare preimage data
        let mut preimages_data = Vec::new();
        let mut preimage_offsets = Vec::new();
        let mut preimage_lengths = Vec::new();

        for preimage in &preimages[..actual_batch_size] {
            preimage_offsets.push(preimages_data.len() as u32);
            let bytes = preimage.as_bytes();
            preimage_lengths.push(bytes.len() as u32);
            preimages_data.extend_from_slice(bytes);
        }

        // Transfer to GPU
        let preimages_gpu = self.device.htod_copy(preimages_data)?;
        let offsets_gpu = self.device.htod_copy(preimage_offsets)?;
        let lengths_gpu = self.device.htod_copy(preimage_lengths)?;

        // Allocate output buffer
        let output_size = actual_batch_size * 64; // 64 bytes per hash
        let mut output_hashes_gpu = self.device.alloc_zeros::<u8>(output_size)?;

        // Configure kernel launch
        let threads_per_block = 256;
        let num_blocks = (actual_batch_size + threads_per_block - 1) / threads_per_block;
        let cfg = LaunchConfig {
            grid_dim: (num_blocks as u32, 1, 1),
            block_dim: (threads_per_block as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        // Launch kernel (clone because launch consumes self)
        unsafe {
            self.kernel_func.clone().launch(
                cfg,
                (
                    &preimages_gpu,
                    &offsets_gpu,
                    &lengths_gpu,
                    &self.rom_data_gpu,
                    &self.rom_digest_gpu,
                    self.rom_size,
                    self.nb_loops,
                    self.nb_instrs,
                    &mut output_hashes_gpu,
                    actual_batch_size as u32,
                )
            )?;
        }

        // Synchronize to ensure kernel completion
        self.device.synchronize()?;

        // Copy results back to host
        let output_hashes = self.device.dtoh_sync_copy(&output_hashes_gpu)?;

        // Convert to hex strings
        let mut results = Vec::new();
        for i in 0..actual_batch_size {
            let hash_bytes = &output_hashes[i * 64..(i + 1) * 64];
            let hash_hex = hex::encode(hash_bytes);
            results.push(hash_hex);
        }

        Ok(results)
    }

    #[cfg(not(feature = "gpu"))]
    pub fn hash_batch(&self, _preimages: &[String]) -> Result<Vec<String>, Box<dyn std::error::Error>> {
        Err("GPU support not compiled".into())
    }

    /// Get GPU device info
    #[cfg(feature = "gpu")]
    pub fn device_info(&self) -> Result<GPUInfo, Box<dyn std::error::Error>> {
        Ok(GPUInfo {
            name: self.device.name()?,
            compute_capability: "6.0+".to_string(),
            total_memory: 0,  // cudarc doesn't expose directly
            free_memory: 0,
        })
    }

    #[cfg(not(feature = "gpu"))]
    pub fn device_info(&self) -> Result<GPUInfo, Box<dyn std::error::Error>> {
        Err("GPU support not compiled".into())
    }

    /// Get batch size
    pub fn batch_size(&self) -> usize {
        self.batch_size
    }

    /// Set ROM parameters (nb_loops and nb_instrs)
    #[cfg(feature = "gpu")]
    pub fn set_rom_params(&mut self, nb_loops: u32, nb_instrs: u32) {
        self.nb_loops = nb_loops;
        self.nb_instrs = nb_instrs;
    }

    #[cfg(not(feature = "gpu"))]
    pub fn set_rom_params(&mut self, _nb_loops: u32, _nb_instrs: u32) {}
}

/// Global GPU engine instance
use std::sync::Mutex;
pub static GPU_ENGINE: Mutex<Option<std::sync::Arc<HashEngineGPU>>> = Mutex::new(None);

/// Initialize global GPU engine
pub fn init_gpu_engine(rom: &Rom, batch_size: usize) -> Result<(), Box<dyn std::error::Error>> {
    if !HashEngineGPU::is_available() {
        return Err("No CUDA GPU available".into());
    }

    let engine = HashEngineGPU::new(rom, batch_size)?;
    let mut global = GPU_ENGINE.lock().unwrap();
    *global = Some(std::sync::Arc::new(engine));

    Ok(())
}

/// Get global GPU engine
pub fn get_gpu_engine() -> Option<std::sync::Arc<HashEngineGPU>> {
    let global = GPU_ENGINE.lock().unwrap();
    global.clone()
}

/// Check if GPU engine is initialized
pub fn is_gpu_initialized() -> bool {
    let global = GPU_ENGINE.lock().unwrap();
    global.is_some()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_available() {
        println!("GPU available: {}", HashEngineGPU::is_available());
    }
}
