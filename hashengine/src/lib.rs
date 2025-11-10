// Import HashEngine modules
mod hashengine;
mod rom;

#[cfg(feature = "gpu")]
pub mod gpu;

use napi::bindgen_prelude::*;
use napi_derive::napi;
use std::sync::Mutex;

use hashengine::hash as sh_hash;
use rom::{RomGenerationType, Rom};

// Global ROM state (similar to ce-ashmaize implementation)
static ROM_STATE: Mutex<Option<Rom>> = Mutex::new(None);
static ROM_READY: Mutex<bool> = Mutex::new(false);
static ROM_PARAMS: Mutex<(u32, u32)> = Mutex::new((8, 256)); // (nb_loops, nb_instrs)

/// Initialize ROM with challenge-specific no_pre_mine value
/// Parameters match the API expectation from TypeScript
///
/// CRITICAL: no_pre_mine_hex is the hex string AS-IS (e.g., "e8a195800b...")
/// We pass it to ROM as UTF-8 bytes, NOT decoded hex bytes
/// This matches HashEngine/src/lib.rs:384 which uses no_pre_mine_key.as_bytes()
#[napi]
pub fn init_rom(
  no_pre_mine_hex: String,
  nb_loops: u32,
  nb_instrs: u32,
  pre_size: u32,
  rom_size: u32,
  mixing_numbers: u32,
) -> Result<()> {
  // CRITICAL: Convert hex STRING to bytes (not decode hex!)
  // This matches HashEngine reference: no_pre_mine_key.as_bytes()
  let no_pre_mine = no_pre_mine_hex.as_bytes();

  // Create ROM using TwoStep generation (matches AshMaze spec)
  let rom = Rom::new(
    no_pre_mine,
    RomGenerationType::TwoStep {
      pre_size: pre_size as usize,
      mixing_numbers: mixing_numbers as usize,
    },
    rom_size as usize,
  );

  // Store ROM in global state
  let mut rom_state = ROM_STATE.lock().unwrap();
  *rom_state = Some(rom);

  let mut ready = ROM_READY.lock().unwrap();
  *ready = true;

  // Store ROM parameters for GPU
  let mut params = ROM_PARAMS.lock().unwrap();
  *params = (nb_loops, nb_instrs);

  Ok(())
}

/// Hash a preimage using HashEngine algorithm
/// Returns 128-char hex string (64 bytes)
#[napi]
pub fn hash_preimage(preimage: String) -> Result<String> {
  // Check ROM is ready
  let ready = ROM_READY.lock().unwrap();
  if !*ready {
    return Err(Error::from_reason("ROM not initialized. Call initRom first."));
  }

  // Get ROM reference
  let rom_state = ROM_STATE.lock().unwrap();
  let rom = rom_state.as_ref()
    .ok_or_else(|| Error::from_reason("ROM not available"))?;

  // Convert preimage string to bytes
  let salt = preimage.as_bytes();

  // Hash using HashEngine (nb_loops=8, nb_instrs=256 per AshMaze spec)
  let hash_bytes = sh_hash(salt, rom, 8, 256);

  // Convert to hex string
  Ok(hex::encode(hash_bytes))
}

/// Check if ROM is ready
#[napi]
pub fn rom_ready() -> bool {
  let ready = ROM_READY.lock().unwrap();
  *ready
}

// ============================================================================
// GPU Mining Functions
// ============================================================================

/// Check if GPU mining is available
#[napi]
pub fn gpu_available() -> bool {
  #[cfg(feature = "gpu")]
  {
    gpu::HashEngineGPU::is_available()
  }
  #[cfg(not(feature = "gpu"))]
  {
    false
  }
}

/// Initialize GPU mining engine
/// Must be called after init_rom
#[napi]
pub fn init_gpu(batch_size: u32) -> Result<()> {
  #[cfg(feature = "gpu")]
  {
    // Get ROM from global state
    let rom_state = ROM_STATE.lock().unwrap();
    let rom = rom_state.as_ref()
      .ok_or_else(|| Error::from_reason("ROM not initialized. Call initRom first."))?;

    // Get ROM parameters
    let params = ROM_PARAMS.lock().unwrap();
    let (nb_loops, nb_instrs) = *params;

    // Initialize GPU engine
    let mut engine = gpu::HashEngineGPU::new(rom, batch_size as usize)
      .map_err(|e| Error::from_reason(format!("GPU initialization failed: {}", e)))?;

    // Set ROM parameters
    engine.set_rom_params(nb_loops, nb_instrs);

    // Store in global state
    let mut global = gpu::GPU_ENGINE.lock().unwrap();
    *global = Some(std::sync::Arc::new(engine));

    Ok(())
  }
  #[cfg(not(feature = "gpu"))]
  {
    Err(Error::from_reason("GPU support not compiled. Rebuild with --features gpu"))
  }
}

/// Check if GPU engine is initialized
#[napi]
pub fn gpu_ready() -> bool {
  #[cfg(feature = "gpu")]
  {
    gpu::is_gpu_initialized()
  }
  #[cfg(not(feature = "gpu"))]
  {
    false
  }
}

/// Hash a batch of preimages on GPU
/// Returns array of hex-encoded hashes
#[napi]
pub fn hash_batch_gpu(preimages: Vec<String>) -> Result<Vec<String>> {
  #[cfg(feature = "gpu")]
  {
    let engine = gpu::get_gpu_engine()
      .ok_or_else(|| Error::from_reason("GPU not initialized. Call initGpu first."))?;

    engine.hash_batch(&preimages)
      .map_err(|e| Error::from_reason(format!("GPU hash batch failed: {}", e)))
  }
  #[cfg(not(feature = "gpu"))]
  {
    Err(Error::from_reason("GPU support not compiled"))
  }
}

/// Get GPU device information
#[napi(object)]
pub struct GpuInfo {
  pub name: String,
  pub compute_capability: String,
  pub total_memory: f64,
  pub free_memory: f64,
}

#[napi]
pub fn gpu_info() -> Result<GpuInfo> {
  #[cfg(feature = "gpu")]
  {
    let engine = gpu::get_gpu_engine()
      .ok_or_else(|| Error::from_reason("GPU not initialized"))?;

    let info = engine.device_info()
      .map_err(|e| Error::from_reason(format!("Failed to get GPU info: {}", e)))?;

    Ok(GpuInfo {
      name: info.name,
      compute_capability: info.compute_capability,
      total_memory: info.total_memory as f64,
      free_memory: info.free_memory as f64,
    })
  }
  #[cfg(not(feature = "gpu"))]
  {
    Err(Error::from_reason("GPU support not compiled"))
  }
}
