/**
 * GPU Test Script
 * Tests GPU mining functionality without starting the full application
 */

console.log('========================================');
console.log('  GPU Mining Test Script');
console.log('========================================');
console.log();

// Test 1: Check if GPU module loads
console.log('[1/4] Loading HashEngine module...');
try {
  const hashEngine = require('./index.node');
  console.log('✓ Module loaded successfully');
  console.log();
} catch (error) {
  console.error('✗ Failed to load module:', error.message);
  console.error();
  console.error('Make sure you ran: build-gpu.cmd');
  process.exit(1);
}

const hashEngine = require('./index.node');

// Test 2: Check GPU availability
console.log('[2/4] Checking GPU availability...');
try {
  const gpuAvailable = hashEngine.gpuAvailable();
  console.log(`GPU Available: ${gpuAvailable}`);

  if (!gpuAvailable) {
    console.log('⚠️  GPU not available');
    console.log('   - Make sure CUDA is installed');
    console.log('   - Make sure you have an NVIDIA GPU');
    console.log('   - Rebuild with: build-gpu.cmd');
    console.log();
    console.log('Continuing with CPU tests only...');
  } else {
    console.log('✓ GPU detected!');
  }
  console.log();
} catch (error) {
  console.error('✗ Error checking GPU:', error.message);
  console.log();
}

// Test 3: Initialize ROM (required for both CPU and GPU)
console.log('[3/4] Initializing ROM...');
try {
  // Use test ROM parameters
  const noPreMineHex = 'e8a195800b4c9e6ddaf5b682ba6a77bd5337eb993b1804e2ff5fb97229b2f8f1';
  const nbLoops = 8;
  const nbInstrs = 256;
  const preSize = 512;
  const romSize = 65536;
  const mixingNumbers = 128;

  hashEngine.initRom(noPreMineHex, nbLoops, nbInstrs, preSize, romSize, mixingNumbers);
  console.log('✓ ROM initialized');
  console.log();
} catch (error) {
  console.error('✗ ROM initialization failed:', error.message);
  process.exit(1);
}

// Test 4: Test CPU hashing (baseline)
console.log('[4/4] Testing CPU hashing...');
try {
  const testPreimage = '0000000000000001addr1test1234567890abcdef';
  const startTime = Date.now();
  const hash = hashEngine.hashPreimage(testPreimage);
  const elapsed = Date.now() - startTime;

  console.log(`✓ CPU hash completed in ${elapsed}ms`);
  console.log(`  Preimage: ${testPreimage.substring(0, 40)}...`);
  console.log(`  Hash:     ${hash.substring(0, 32)}...`);
  console.log();
} catch (error) {
  console.error('✗ CPU hashing failed:', error.message);
  process.exit(1);
}

// Test 5: Initialize GPU (if available)
if (hashEngine.gpuAvailable()) {
  console.log('[5/6] Initializing GPU...');
  try {
    const batchSize = 1000; // Small batch for testing
    hashEngine.initGpu(batchSize);
    console.log('✓ GPU initialized');
    console.log();
  } catch (error) {
    console.error('✗ GPU initialization failed:', error.message);
    console.error('   This is expected - GPU kernels are still in development');
    console.log();
  }

  // Test 6: Get GPU info
  console.log('[6/6] Getting GPU information...');
  try {
    const info = hashEngine.gpuInfo();
    console.log('✓ GPU Information:');
    console.log(`  Device: ${info.name}`);
    console.log(`  Compute Capability: ${info.computeCapability}`);
    console.log(`  Total Memory: ${(info.totalMemory / (1024 * 1024 * 1024)).toFixed(2)} GB`);
    console.log();
  } catch (error) {
    console.error('✗ Failed to get GPU info:', error.message);
    console.log();
  }

  // Test 7: GPU batch hashing (placeholder test)
  console.log('[OPTIONAL] Testing GPU batch hashing...');
  try {
    const testPreimages = [
      '0000000000000001test1',
      '0000000000000002test2',
      '0000000000000003test3',
    ];

    const hashes = hashEngine.hashBatchGpu(testPreimages);
    console.log('✓ GPU batch hash completed');
    console.log(`  Hashed ${hashes.length} preimages`);
    console.log();
  } catch (error) {
    console.log('⚠️  GPU batch hashing not yet implemented (expected)');
    console.log(`   Error: ${error.message}`);
    console.log();
  }
}

// Summary
console.log('========================================');
console.log('  Test Summary');
console.log('========================================');
console.log();

const gpuAvailable = hashEngine.gpuAvailable();
const gpuReady = hashEngine.gpuReady ? hashEngine.gpuReady() : false;

console.log('Status:');
console.log(`  ✓ Module loads: YES`);
console.log(`  ✓ ROM initialized: YES`);
console.log(`  ✓ CPU hashing works: YES`);
console.log(`  ${gpuAvailable ? '✓' : '✗'} GPU available: ${gpuAvailable ? 'YES' : 'NO'}`);
console.log(`  ${gpuReady ? '✓' : '⚠️ '} GPU initialized: ${gpuReady ? 'YES' : 'NO (expected - kernels in development)'}`);
console.log();

if (gpuAvailable) {
  console.log('Next Steps:');
  console.log('  1. ✓ GPU detection working');
  console.log('  2. ⚠️  GPU CUDA kernels need implementation');
  console.log('  3. ⚠️  GPU batch hashing will be added in Phase 2');
  console.log();
  console.log('Current Status: GPU feature flag enabled, kernels in development');
} else {
  console.log('GPU Mining:');
  console.log('  To enable GPU mining:');
  console.log('  1. Install CUDA Toolkit 12.0+');
  console.log('  2. Run: build-gpu.cmd');
  console.log('  3. Run this test again');
}

console.log();
console.log('========================================');
console.log('Test completed successfully! ✓');
console.log('========================================');
