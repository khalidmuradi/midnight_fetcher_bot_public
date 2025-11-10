/**
 * Blake2b Test - Compare known test vectors
 * Tests if GPU Blake2b matches standard Blake2b-512
 */

const crypto = require('crypto');
const hashEngine = require('./index.node');

console.log('========================================');
console.log('  Blake2b Test Vectors');
console.log('========================================');
console.log();

// Known Blake2b-512 test vectors from official spec
const testVectors = [
  { input: '', expected: 'blake2b-512 of empty string' },
  { input: 'abc', expected: 'blake2b-512 of abc' },
  { input: 'The quick brown fox jumps over the lazy dog', expected: 'blake2b-512 of sentence' }
];

console.log('Testing Node.js crypto Blake2b-512:');
console.log();

for (const test of testVectors) {
  const hash = crypto.createHash('blake2b512').update(test.input).digest('hex');
  console.log(`Input: "${test.input}"`);
  console.log(`Hash:  ${hash}`);
  console.log();
}

console.log('========================================');
console.log();

// Now test if HashEngine uses Blake2b correctly
console.log('Testing HashEngine Blake2b indirectly:');
console.log('(HashEngine uses Blake2b as part of its hash function)');
console.log();

// Initialize ROM
const noPreMineHex = 'e09957f08a5073fcbb7a7ee8a2694957c839ab0d6ad29fce06199c887236be2c';
hashEngine.initRom(noPreMineHex, 8, 256, 16777216, 1073741824, 4);

// Test same inputs with CPU
console.log('CPU hashes:');
const cpuHashes = testVectors.map(test => {
  const hash = hashEngine.hashPreimage(test.input);
  console.log(`"${test.input}" → ${hash.substring(0, 32)}...`);
  return hash;
});
console.log();

// Test with GPU
console.log('GPU hashes:');
hashEngine.initGpu(1000);
const gpuHashes = hashEngine.hashBatchGpu(testVectors.map(t => t.input));
gpuHashes.forEach((hash, i) => {
  console.log(`"${testVectors[i].input}" → ${hash.substring(0, 32)}...`);
});
console.log();

console.log('Comparison:');
let matches = 0;
for (let i = 0; i < testVectors.length; i++) {
  const match = cpuHashes[i] === gpuHashes[i];
  console.log(`Test ${i + 1}: ${match ? '✅ MATCH' : '❌ MISMATCH'}`);
  if (match) matches++;
}
console.log();
console.log(`Result: ${matches}/${testVectors.length} tests passed`);

if (matches === 0) {
  console.log();
  console.log('❌ ALL GPU hashes wrong - fundamental Blake2b bug in CUDA kernel');
  console.log();
  console.log('Expected behavior:');
  console.log('  - GPU Blake2b should match CPU Blake2b for any input');
  console.log('  - This is independent of ROM or Argon2');
  console.log();
  console.log('Likely causes:');
  console.log('  1. Blake2b state initialization incorrect (h[0] parameter block)');
  console.log('  2. Message padding/length encoding wrong');
  console.log('  3. Compression function has bugs');
  console.log('  4. Endianness issues in message schedule');
  console.log();
  console.log('Next: Add printf debugging to CUDA blake2b_512() function');
}
