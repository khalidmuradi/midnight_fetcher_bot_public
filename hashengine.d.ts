/**
 * TypeScript declarations for HashEngine NAPI module
 * Auto-generated from Rust NAPI bindings
 */

declare module '@/hashengine' {
  /**
   * GPU device information
   */
  export interface GpuInfo {
    name: string;
    compute_capability: string;
    total_memory: number;
    free_memory: number;
  }

  /**
   * Initialize ROM with challenge-specific parameters
   * Must be called before any hashing operations
   */
  export function initRom(
    noPreMineHex: string,
    nbLoops: number,
    nbInstrs: number,
    preSize: number,
    romSize: number,
    mixingNumbers: number
  ): void;

  /**
   * Check if ROM is initialized and ready
   */
  export function romReady(): boolean;

  /**
   * Hash a single preimage using CPU HashEngine
   * Returns 128-char hex string (64 bytes)
   */
  export function hashPreimage(preimage: string): string;

  /**
   * Check if GPU mining is available on this system
   */
  export function gpuAvailable(): boolean;

  /**
   * Initialize GPU mining engine
   * Must be called after initRom
   * @param batchSize - Number of hashes to process per GPU batch
   */
  export function initGpu(batchSize: number): void;

  /**
   * Check if GPU engine is initialized and ready
   */
  export function gpuReady(): boolean;

  /**
   * Hash a batch of preimages on GPU (CUDA accelerated)
   * Much faster than CPU for large batches
   * @param preimages - Array of preimage strings to hash
   * @returns Array of hex-encoded hash strings
   */
  export function hashBatchGpu(preimages: string[]): string[];

  /**
   * Get GPU device information
   * Only available after GPU is initialized
   */
  export function gpuInfo(): GpuInfo;
}
