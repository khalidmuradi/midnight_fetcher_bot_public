/**
 * Mining Configuration
 * Controls CPU and GPU mining parameters
 */

export interface MiningConfig {
  // CPU settings
  cpuWorkers: number;
  cpuBatchSize: number;

  // GPU settings
  enableGpu: boolean;
  gpuBatchSize: number;
  gpuWorkers: number;  // Number of parallel GPU batches

  // Nonce range allocation
  cpuNonceStart: bigint;
  cpuNonceEnd: bigint;
  gpuNonceStart: bigint;
  gpuNonceEnd: bigint;
}

export const DEFAULT_MINING_CONFIG: MiningConfig = {
  // CPU: 11 workers, 300 hashes per batch
  cpuWorkers: 11,  // ✅ CPU mining works correctly
  cpuBatchSize: 300,

  // GPU: DISABLED - implementation has hash computation bugs
  enableGpu: false,  // ❌ GPU produces incorrect hashes - needs complete rewrite
  gpuBatchSize: 150000,
  gpuWorkers: 4,

  // Nonce ranges (prevent overlap)
  // CPU gets 0 to 10 billion
  cpuNonceStart: 0n,
  cpuNonceEnd: 10_000_000_000n,

  // GPU gets 10 billion to 1 trillion
  gpuNonceStart: 10_000_000_000n,
  gpuNonceEnd: 1_000_000_000_000n,
};

class MiningConfigManager {
  private config: MiningConfig = { ...DEFAULT_MINING_CONFIG };

  getConfig(): MiningConfig {
    return { ...this.config };
  }

  updateConfig(updates: Partial<MiningConfig>): void {
    this.config = { ...this.config, ...updates };
    console.log('[MiningConfig] Configuration updated:', this.config);
  }

  // Helper: Get nonce range for a specific CPU worker
  getCpuWorkerNonceRange(workerId: number, totalWorkers: number): { start: bigint; end: bigint } {
    const rangeSize = this.config.cpuNonceEnd - this.config.cpuNonceStart;
    const workerRange = rangeSize / BigInt(totalWorkers);

    return {
      start: this.config.cpuNonceStart + (BigInt(workerId) * workerRange),
      end: this.config.cpuNonceStart + (BigInt(workerId + 1) * workerRange),
    };
  }

  // Helper: Get nonce range for a specific GPU worker
  getGpuWorkerNonceRange(workerId: number, totalWorkers: number): { start: bigint; end: bigint } {
    const rangeSize = this.config.gpuNonceEnd - this.config.gpuNonceStart;
    const workerRange = rangeSize / BigInt(totalWorkers);

    return {
      start: this.config.gpuNonceStart + (BigInt(workerId) * workerRange),
      end: this.config.gpuNonceStart + (BigInt(workerId + 1) * workerRange),
    };
  }
}

export const miningConfig = new MiningConfigManager();
