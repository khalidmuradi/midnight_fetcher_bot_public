/**
 * GPU Kernel Debugging Test
 * Tests individual components to find where GPU diverges from CPU
 */

const hashEngine = require('./index.node');

console.log('========================================');
console.log('  GPU Kernel Component Debug Test');
console.log('========================================');
console.log();

// Initialize ROM
const noPreMineHex = 'e09957f08a5073fcbb7a7ee8a2694957c839ab0d6ad29fce06199c887236be2c';
const nbLoops = 8;
const nbInstrs = 256;
const preSize = 16777216;
const romSize = 1073741824;
const mixingNumbers = 4;

console.log('[1/4] Initializing ROM...');
hashEngine.initRom(noPreMineHex, nbLoops, nbInstrs, preSize, romSize, mixingNumbers);
console.log('✓ ROM initialized');
console.log();

// Test with multiple simple preimages to isolate pattern
const testCases = [
  'test1',
  'test2',
  'hello',
  'a',
  'ab',
  'abc',
  '0000003bf44f3623addr1qxf9chy7euve40tvhdtuxnfx3dsxe2366k96z53v23nr9x2lrth6adf2a4thwacp5wd39lz2808yfqranc2k3sepw4hsap8fs4**D10C24000007FFe09957f08a5073fcbb7a7ee8a2694957c839ab0d6ad29fce06199c887236be2c2025-11-09T22:59:59.000Z287460848'
];

console.log('[2/4] Initializing GPU...');
hashEngine.initGpu(1000);
console.log('✓ GPU initialized');
console.log();

console.log('[3/4] Testing hash consistency across multiple inputs...');
console.log();

let passCount = 0;
let failCount = 0;

for (let i = 0; i < testCases.length; i++) {
  const preimage = testCases[i];

  console.log(`Test ${i + 1}/${testCases.length}: "${preimage.substring(0, 40)}${preimage.length > 40 ? '...' : ''}"`);

  const cpuHash = hashEngine.hashPreimage(preimage);
  const gpuHashes = hashEngine.hashBatchGpu([preimage]);
  const gpuHash = gpuHashes[0];

  const match = cpuHash === gpuHash;

  if (match) {
    console.log('  ✅ MATCH');
    passCount++;
  } else {
    console.log('  ❌ MISMATCH');
    console.log('  CPU:', cpuHash.substring(0, 64) + '...');
    console.log('  GPU:', gpuHash.substring(0, 64) + '...');

    // Check if first 8 chars match (Blake2b initialization)
    if (cpuHash.substring(0, 8) === gpuHash.substring(0, 8)) {
      console.log('  → First 8 chars match - Blake2b init likely OK');
    } else {
      console.log('  → First 8 chars differ - Blake2b issue');
    }

    // Check if ANY part matches
    let anyMatch = false;
    for (let j = 0; j < 64; j += 8) {
      if (cpuHash.substring(j, j + 8) === gpuHash.substring(j, j + 8)) {
        anyMatch = true;
        console.log(`  → Chars ${j}-${j + 8} match`);
        break;
      }
    }
    if (!anyMatch) {
      console.log('  → No overlap at all - fundamental issue');
    }

    failCount++;
  }
  console.log();
}

console.log('[4/4] Summary');
console.log('========================================');
console.log(`Passed: ${passCount}/${testCases.length}`);
console.log(`Failed: ${failCount}/${testCases.length}`);
console.log();

if (failCount === testCases.length) {
  console.log('❌ ALL TESTS FAILED');
  console.log();
  console.log('Diagnosis: GPU kernels have fundamental implementation bug');
  console.log('Likely causes:');
  console.log('  1. Blake2b-512 kernel has incorrect state initialization or compression');
  console.log('  2. Endianness mismatch (little-endian vs big-endian)');
  console.log('  3. Padding or length encoding incorrect');
  console.log('  4. Argon2 hprime computation wrong');
  console.log('  5. VM instruction execution differs from CPU');
  console.log();
  console.log('Next steps:');
  console.log('  1. Add printf debugging to CUDA kernels');
  console.log('  2. Compare Blake2b state after initialization');
  console.log('  3. Test Blake2b with known test vectors');
  console.log('  4. Validate Argon2 output against CPU');
} else if (failCount > 0) {
  console.log('⚠️  PARTIAL FAILURES');
  console.log();
  console.log('Some inputs work, others fail - may be length-dependent bug');
} else {
  console.log('✅ ALL TESTS PASSED');
  console.log();
  console.log('GPU kernels appear to be working correctly!');
  console.log('Previous failures may have been configuration issues.');
}
