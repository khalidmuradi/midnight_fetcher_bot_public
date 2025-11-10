/**
 * GPU Hash Engine Wrapper
 * Provides TypeScript interface to CUDA-accelerated HashEngine
 */

export interface GpuInfo {
  name: string;
  compute_capability: string;
  total_memory: number;
  free_memory: number;
}

// Cache for the native module (loaded lazily at runtime)
let _nativeModuleCache: any = undefined;

// Safely load native module - only available server-side
function getNativeModule() {
  // Return cached module if already loaded
  if (_nativeModuleCache !== undefined) {
    return _nativeModuleCache;
  }

  try {
    // Only load on server side (Node.js environment)
    if (typeof window === 'undefined') {
      // Use eval to prevent Turbopack from analyzing this require() at build time
      const requireFunc = eval('require');
      const path = requireFunc('path');
      const modulePath = path.join(process.cwd(), 'index.node');
      _nativeModuleCache = requireFunc(modulePath);
      return _nativeModuleCache;
    }
  } catch (error) {
    console.warn('[HashEngineGPU] Native module failed to load:', error);
    _nativeModuleCache = null;
  }

  _nativeModuleCache = null;
  return null;
}

class HashEngineGPU {
  private initialized = false;
  private batchSize = 10000; // Default: 10K hashes per GPU batch

  /**
   * Check if GPU mining is available on this system
   */
  isAvailable(): boolean {
    const nativeModule = getNativeModule();
    if (!nativeModule || typeof nativeModule.gpuAvailable !== 'function') {
      return false;
    }

    try {
      return nativeModule.gpuAvailable();
    } catch (error) {
      console.error('[HashEngineGPU] Error checking GPU availability:', error);
      return false;
    }
  }

  /**
   * Initialize ROM in native module
   * Must be called before GPU initialization
   */
  async initRom(noPreMineHex: string): Promise<void> {
    const nativeModule = getNativeModule();
    if (!nativeModule) {
      throw new Error('Native module not loaded');
    }

    // Initialize ROM with same parameters as hash server
    // These are the Midnight network standard parameters
    nativeModule.initRom(
      noPreMineHex,
      8,      // nb_loops
      256,    // nb_instrs
      4096,   // pre_size
      65536,  // rom_size
      64      // mixing_numbers
    );

    console.log('[HashEngineGPU] ROM initialized in native module');
  }

  /**
   * Check if ROM is ready in native module
   */
  isRomReady(): boolean {
    const nativeModule = getNativeModule();
    if (!nativeModule || typeof nativeModule.romReady !== 'function') {
      return false;
    }

    try {
      return nativeModule.romReady();
    } catch (error) {
      return false;
    }
  }

  /**
   * Initialize GPU mining engine
   * Must be called after ROM is initialized
   */
  async init(batchSize?: number): Promise<void> {
    if (!this.isAvailable()) {
      throw new Error('GPU mining not available on this system');
    }

    if (this.initialized) {
      console.log('[HashEngineGPU] Already initialized');
      return;
    }

    // Check ROM is ready
    if (!this.isRomReady()) {
      throw new Error('ROM not initialized. Call initRom first.');
    }

    if (batchSize) {
      this.batchSize = batchSize;
    }

    console.log(`[HashEngineGPU] Initializing with batch size: ${this.batchSize}`);

    try {
      const nativeModule = getNativeModule();
      if (!nativeModule) {
        throw new Error('Native module not loaded');
      }

      nativeModule.initGpu(this.batchSize);
      this.initialized = true;

      // Get and log GPU info
      const info = this.getInfo();
      console.log('[HashEngineGPU] ========================================');
      console.log('[HashEngineGPU] GPU Mining Initialized');
      console.log(`[HashEngineGPU] Device: ${info.name}`);
      console.log(`[HashEngineGPU] Compute: ${info.compute_capability}`);
      console.log(`[HashEngineGPU] Memory: ${(info.total_memory / (1024 * 1024 * 1024)).toFixed(2)} GB`);
      console.log(`[HashEngineGPU] Batch Size: ${this.batchSize}`);
      console.log('[HashEngineGPU] ========================================');
    } catch (error: any) {
      throw new Error(`Failed to initialize GPU: ${error.message}`);
    }
  }

  /**
   * Check if GPU engine is initialized and ready
   */
  isReady(): boolean {
    const nativeModule = getNativeModule();
    if (!nativeModule || typeof nativeModule.gpuReady !== 'function') {
      return false;
    }

    try {
      return nativeModule.gpuReady();
    } catch (error) {
      return false;
    }
  }

  /**
   * Hash a batch of preimages on GPU
   * Much faster than CPU for large batches
   */
  async hashBatch(preimages: string[]): Promise<string[]> {
    if (!this.isReady()) {
      throw new Error('GPU not initialized. Call init() first.');
    }

    if (preimages.length === 0) {
      return [];
    }

    try {
      const nativeModule = getNativeModule();
      if (!nativeModule) {
        throw new Error('Native module not loaded');
      }

      const startTime = Date.now();
      const hashes = nativeModule.hashBatchGpu(preimages);
      const elapsed = Date.now() - startTime;

      const hashRate = Math.round((preimages.length / elapsed) * 1000);
      console.log(`[HashEngineGPU] Hashed ${preimages.length} preimages in ${elapsed}ms (${hashRate.toLocaleString()} H/s)`);

      return hashes;
    } catch (error: any) {
      throw new Error(`GPU batch hash failed: ${error.message}`);
    }
  }

  /**
   * Get GPU device information
   */
  getInfo(): GpuInfo {
    if (!this.isReady()) {
      throw new Error('GPU not initialized');
    }

    const nativeModule = getNativeModule();
    if (!nativeModule) {
      throw new Error('Native module not loaded');
    }

    return nativeModule.gpuInfo();
  }

  /**
   * Get current batch size
   */
  getBatchSize(): number {
    return this.batchSize;
  }

  /**
   * Update batch size (requires re-initialization)
   */
  async setBatchSize(newBatchSize: number): Promise<void> {
    this.batchSize = newBatchSize;
    if (this.initialized) {
      console.log(`[HashEngineGPU] Batch size changed to ${newBatchSize}, reinitializing...`);
      this.initialized = false;
      await this.init();
    }
  }
}

// Singleton instance
export const hashEngineGPU = new HashEngineGPU();
