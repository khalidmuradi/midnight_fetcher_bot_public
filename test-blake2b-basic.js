/**
 * Test basic Blake2b against Node.js crypto
 * This will help us understand what's different
 */

const crypto = require('crypto');
const hashEngine = require('./index.node');

console.log('Testing if Blake2b is being used correctly...');
console.log();

// Test 1: Raw Blake2b-512 on "abc"
const node_blake2b_abc = crypto.createHash('blake2b512').update('abc').digest('hex');
console.log('Node.js Blake2b-512("abc"):', node_blake2b_abc);
console.log();

// Test 2: HashEngine on "abc"
hashEngine.initRom('e09957f08a5073fcbb7a7ee8a2694957c839ab0d6ad29fce06199c887236be2c', 8, 256, 16777216, 1073741824, 4);
const cpu_hash_abc = hashEngine.hashPreimage('abc');
console.log('CPU HashEngine("abc"):', cpu_hash_abc);
console.log();

hashEngine.initGpu(100);
const gpu_hash_abc = hashEngine.hashBatchGpu(['abc'])[0];
console.log('GPU HashEngine("abc"):', gpu_hash_abc);
console.log();

console.log('Analysis:');
console.log('- Node Blake2b != CPU HashEngine (expected - HashEngine does more than just Blake2b)');
console.log('- CPU HashEngine should == GPU HashEngine (they should do the same thing!)');
console.log();

if (cpu_hash_abc === gpu_hash_abc) {
    console.log('✅ GPU MATCHES CPU!');
} else {
    console.log('❌ GPU DIFFERS FROM CPU');
    console.log();
    console.log('This means the GPU implementation has a bug in:');
    console.log('  - Blake2b-512 implementation, OR');
    console.log('  - Argon2 hprime implementation, OR');
    console.log('  - VM execution logic');
    console.log();

    // Compare first few chars to see where divergence starts
    for (let i = 0; i < 128; i += 8) {
        const cpuChunk = cpu_hash_abc.substring(i, i + 8);
        const gpuChunk = gpu_hash_abc.substring(i, i + 8);
        if (cpuChunk !== gpuChunk) {
            console.log(`First difference at char ${i}:`);
            console.log(`  CPU: ${cpuChunk}`);
            console.log(`  GPU: ${gpuChunk}`);
            break;
        }
    }
}
