/**
 * Check GPU Mining Status
 * Shows what logs to expect and current GPU state
 */

const engine = require('./index.node');

console.log('========================================');
console.log('GPU Mining Status Check');
console.log('========================================\n');

// Check current build
console.log('[Build Info]');
console.log('GPU functions available:', {
  gpuAvailable: typeof engine.gpuAvailable === 'function',
  initGpu: typeof engine.initGpu === 'function',
  gpuReady: typeof engine.gpuReady === 'function',
  hashBatchGpu: typeof engine.hashBatchGpu === 'function',
});
console.log();

// Check GPU availability
console.log('[GPU Hardware]');
const isAvailable = engine.gpuAvailable();
console.log('GPU Available:', isAvailable);

if (isAvailable) {
  console.log('✅ CUDA GPU detected');
  console.log('✅ GPU mining will activate automatically when mining starts');
} else {
  console.log('❌ No CUDA GPU detected');
  console.log('   GPU mining will NOT activate');
}
console.log();

// Show expected logs
console.log('========================================');
console.log('Expected Logs When Mining Starts');
console.log('========================================\n');

console.log('1️⃣  When a new challenge starts, look for:');
console.log('   [Orchestrator] ========================================');
console.log('   [Orchestrator] MINING ENGINE STATUS');
console.log('   [Orchestrator] ========================================');
console.log('   [Orchestrator] CPU Mining: ENABLED (11 workers)');
if (isAvailable) {
  console.log('   [Orchestrator] GPU Mining: ENABLED (4 workers)  ⬅️  LOOK FOR THIS');
  console.log('   [Orchestrator] GPU Batch Size: 100,000');
  console.log('   [Orchestrator] Expected speedup: ~100-200x vs CPU');
} else {
  console.log('   [Orchestrator] GPU Mining: DISABLED');
}
console.log();

console.log('2️⃣  When workers launch for an address:');
console.log('   [Orchestrator] ========================================');
console.log('   [Orchestrator] LAUNCHING MINING WORKERS');
console.log('   [Orchestrator] ========================================');
console.log('   [Orchestrator] CPU Workers: 8 (0-7)');
if (isAvailable) {
  console.log('   [Orchestrator] GPU Workers: 4 (1000-1003)  ⬅️  LOOK FOR THIS');
  console.log('   [Orchestrator] ⚡ GPU MINING ACTIVE - Expect 100-200x speedup!');
}
console.log();

console.log('3️⃣  GPU worker startup (one per GPU worker):');
if (isAvailable) {
  console.log('   [Orchestrator] ========================================');
  console.log('   [Orchestrator] [GPU] Worker 0 STARTED  ⬅️  LOOK FOR THIS');
  console.log('   [Orchestrator] ========================================');
  console.log('   [Orchestrator] [GPU] Address: tnight1...');
  console.log('   [Orchestrator] [GPU] Batch Size: 100,000 hashes/batch');
  console.log('   [Orchestrator] [GPU] Nonce Range: 10,000,000,000 to ...');
}
console.log();

console.log('========================================');
console.log('Where to Find These Logs');
console.log('========================================\n');

console.log('🔍 In the NEXT.JS console (NOT hash-server console)');
console.log('   When you run: npm start');
console.log();

console.log('⚠️  GPU logs only appear when:');
console.log('   1. A new challenge is detected');
console.log('   2. Workers are launched to mine an address');
console.log();

if (!isAvailable) {
  console.log('❌ GPU NOT AVAILABLE - logs will show "GPU Mining: DISABLED"');
  console.log('   To enable GPU:');
  console.log('   1. Install CUDA Toolkit 12.0+');
  console.log('   2. Run: build-gpu.cmd');
  console.log('   3. Restart mining');
} else {
  console.log('✅ GPU IS AVAILABLE - logs will show GPU activity when mining starts');
  console.log();
  console.log('Next steps:');
  console.log('   1. Start hash server: cd hashengine/target/release && ./hash-server.exe');
  console.log('   2. Start Next.js: npm start');
  console.log('   3. Start mining from the web UI');
  console.log('   4. Watch the Next.js console for GPU logs');
}
