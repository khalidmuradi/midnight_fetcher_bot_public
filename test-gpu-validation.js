/**
 * GPU Validation Test
 * Tests if GPU produces same hashes as CPU for identical inputs
 */

const hashEngine = require('./index.node');

console.log('========================================');
console.log('  GPU Hash Validation Test');
console.log('========================================');
console.log();

// Test parameters from your challenge
const noPreMineHex = 'e09957f08a5073fcbb7a7ee8a2694957c839ab0d6ad29fce06199c887236be2c';
const nbLoops = 8;
const nbInstrs = 256;
const preSize = 16777216;
const romSize = 1073741824;
const mixingNumbers = 4;

console.log('[1/5] Initializing ROM...');
hashEngine.initRom(noPreMineHex, nbLoops, nbInstrs, preSize, romSize, mixingNumbers);
console.log('✓ ROM initialized');
console.log();

// Test preimage from your logs
const testPreimage = '0000003bf44f3623addr1qxf9chy7euve40tvhdtuxnfx3dsxe2366k96z53v23nr9x2lrth6adf2a4thwacp5wd39lz2808yfqranc2k3sepw4hsap8fs4**D10C24000007FFe09957f08a5073fcbb7a7ee8a2694957c839ab0d6ad29fce06199c887236be2c2025-11-09T22:59:59.000Z287460848';

console.log('[2/5] Testing CPU hash...');
console.log('Preimage:', testPreimage.substring(0, 80) + '...');
const cpuStart = Date.now();
const cpuHash = hashEngine.hashPreimage(testPreimage);
const cpuTime = Date.now() - cpuStart;
console.log('CPU Hash:', cpuHash);
console.log('CPU Time:', cpuTime + 'ms');
console.log();

console.log('[3/5] Checking GPU availability...');
const gpuAvailable = hashEngine.gpuAvailable();
console.log('GPU Available:', gpuAvailable);

if (!gpuAvailable) {
  console.log('❌ GPU not available. Rebuild with: build-gpu.cmd');
  process.exit(1);
}
console.log();

console.log('[4/5] Initializing GPU...');
const batchSize = 1000;
try {
  hashEngine.initGpu(batchSize);
  console.log('✓ GPU initialized');
} catch (e) {
  console.log('❌ GPU initialization failed:', e.message);
  process.exit(1);
}
console.log();

console.log('[5/5] Testing GPU hash (batch of 1)...');
const gpuStart = Date.now();
try {
  const gpuHashes = hashEngine.hashBatchGpu([testPreimage]);
  const gpuTime = Date.now() - gpuStart;
  const gpuHash = gpuHashes[0];

  console.log('GPU Hash:', gpuHash);
  console.log('GPU Time:', gpuTime + 'ms');
  console.log();

  console.log('========================================');
  console.log('  VALIDATION RESULTS');
  console.log('========================================');
  console.log('CPU Hash:', cpuHash);
  console.log('GPU Hash:', gpuHash);
  console.log();

  if (cpuHash === gpuHash) {
    console.log('✅ PASS - GPU and CPU produce identical hashes!');
    console.log();

    // Count leading zero bits
    const leadingZeroBits = countLeadingZeroBits(cpuHash);
    console.log('Leading zero bits:', leadingZeroBits);
    console.log('Required:', 21);
    console.log('Meets difficulty?', leadingZeroBits >= 21 ? '✅ YES' : '❌ NO');
  } else {
    console.log('❌ FAIL - GPU produces DIFFERENT hash than CPU!');
    console.log();
    console.log('This means the CUDA kernel has bugs.');
    console.log('GPU implementation needs debugging.');
  }
} catch (e) {
  console.log('❌ GPU hash failed:', e.message);
  console.log();
  console.log('GPU kernel may not be fully implemented yet.');
}
console.log();

function countLeadingZeroBits(hexHash) {
  let count = 0;
  for (let i = 0; i < hexHash.length; i++) {
    const nibble = parseInt(hexHash[i], 16);
    if (nibble === 0) {
      count += 4;
    } else {
      // Count leading zeros in this nibble
      count += Math.clz32(nibble) - 28; // 32-bit clz, adjust for 4-bit nibble
      break;
    }
  }
  return count;
}
