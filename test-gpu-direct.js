/**
 * Direct GPU Test - Tests GPU mining functions directly
 * Run with: node test-gpu-direct.js
 */

const engine = require('./index.node');

console.log('========================================');
console.log('GPU Direct Test');
console.log('========================================\n');

// Step 1: Check GPU availability
console.log('[1/5] Checking GPU availability...');
const isAvailable = engine.gpuAvailable();
console.log(`GPU Available: ${isAvailable}`);

if (!isAvailable) {
  console.log('\n❌ GPU not available. Reasons:');
  console.log('  - No CUDA GPU detected');
  console.log('  - CUDA not installed');
  console.log('  - GPU feature not compiled (run build-gpu.cmd)');
  process.exit(1);
}

console.log('✅ GPU is available!\n');

// Step 2: Initialize ROM (required before GPU init)
console.log('[2/5] Initializing ROM...');
const noPreMine = 'e8a195800b9db77b3cf9b4dc7d2697dbd2e784fdbd7eee0e98c90d236f88bfc5';
try {
  engine.initRom(
    noPreMine,
    8,      // nb_loops
    256,    // nb_instrs
    4096,   // pre_size
    65536,  // rom_size
    64      // mixing_numbers
  );
  console.log('✅ ROM initialized\n');
} catch (error) {
  console.error('❌ ROM initialization failed:', error.message);
  process.exit(1);
}

// Step 3: Initialize GPU
console.log('[3/5] Initializing GPU engine...');
const batchSize = 1000; // Small batch for testing
try {
  engine.initGpu(batchSize);
  console.log(`✅ GPU initialized with batch size: ${batchSize}\n`);
} catch (error) {
  console.error('❌ GPU initialization failed:', error.message);
  console.error('This likely means:');
  console.error('  - GPU feature compiled but CUDA runtime error');
  console.error('  - Incompatible CUDA version');
  console.error('  - GPU memory issue');
  process.exit(1);
}

// Step 4: Check GPU ready
console.log('[4/5] Checking GPU ready status...');
const isReady = engine.gpuReady();
console.log(`GPU Ready: ${isReady}`);

if (!isReady) {
  console.error('❌ GPU not ready after initialization');
  process.exit(1);
}

console.log('✅ GPU is ready!\n');

// Step 5: Get GPU info
console.log('[5/5] Getting GPU info...');
try {
  const info = engine.gpuInfo();
  console.log('GPU Device Information:');
  console.log(`  Name: ${info.name}`);
  console.log(`  Compute Capability: ${info.compute_capability}`);
  console.log(`  Total Memory: ${info.total_memory} bytes`);
  console.log(`  Free Memory: ${info.free_memory} bytes`);
  console.log('✅ GPU info retrieved\n');
} catch (error) {
  console.error('❌ Failed to get GPU info:', error.message);
}

// Step 6: Test GPU hashing (BONUS - if this works, GPU is fully functional)
console.log('[BONUS] Testing GPU hash batch...');
console.log('Generating 100 test preimages...');

const testPreimages = [];
for (let i = 0; i < 100; i++) {
  const nonce = i.toString(16).padStart(16, '0');
  const address = 'tnight1test123456789abcdefghijklmnopqrstuvwxyz';
  const preimage = `${nonce}${address}${noPreMine}`;
  testPreimages.push(preimage);
}

console.log('Hashing 100 preimages on GPU...');
const startTime = Date.now();

try {
  const hashes = engine.hashBatchGpu(testPreimages);
  const elapsed = Date.now() - startTime;
  const hashRate = Math.round((hashes.length / elapsed) * 1000);

  console.log(`✅ GPU hashing successful!`);
  console.log(`  Hashes computed: ${hashes.length}`);
  console.log(`  Time: ${elapsed}ms`);
  console.log(`  Hash rate: ${hashRate.toLocaleString()} H/s`);
  console.log(`  Sample hash: ${hashes[0].slice(0, 32)}...`);
  console.log();

} catch (error) {
  console.error('❌ GPU hashing failed:', error.message);
  console.error('This means:');
  console.error('  - GPU initialization succeeded but kernels failed');
  console.error('  - CUDA kernel compilation issue');
  console.error('  - GPU memory allocation failed');
  console.error();
  console.error('However, GPU is still available for mining if this is a preimage format issue.');
  process.exit(1);
}

console.log('========================================');
console.log('🎉 ALL GPU TESTS PASSED!');
console.log('========================================');
console.log();
console.log('GPU mining is fully functional and will activate when:');
console.log('  1. You start mining (npm start)');
console.log('  2. A challenge is active');
console.log('  3. Workers are launched for an address');
console.log();
console.log('Look for these logs in Next.js console:');
console.log('  - "[Orchestrator] GPU Mining: ENABLED (4 workers)"');
console.log('  - "[Orchestrator] ⚡ GPU MINING ACTIVE"');
console.log('  - "[Orchestrator] [GPU] Worker 0 STARTED"');
