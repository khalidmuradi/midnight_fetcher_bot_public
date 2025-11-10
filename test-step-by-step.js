/**
 * Step-by-step comparison of CPU vs GPU
 * This will help us find EXACTLY where they diverge
 */

const crypto = require('crypto');
const hashEngine = require('./index.node');

console.log('====================================');
console.log('Step-by-Step Blake2b Debugging');
console.log('====================================');
console.log();

// Initialize
const noPreMineHex = 'e09957f08a5073fcbb7a7ee8a2694957c839ab0d6ad29fce06199c887236be2c';
hashEngine.initRom(noPreMineHex, 8, 256, 16777216, 1073741824, 4);
hashEngine.initGpu(100);

console.log('Test 1: Very simple input - "abc"');
console.log('-----------------------------------');
const cpu1 = hashEngine.hashPreimage('abc');
const gpu1 = hashEngine.hashBatchGpu(['abc'])[0];
console.log('CPU:', cpu1.substring(0, 64));
console.log('GPU:', gpu1.substring(0, 64));
console.log('Match:', cpu1 === gpu1 ? '✅' : '❌');
console.log();

console.log('Test 2: Single byte - "a"');
console.log('-----------------------------------');
const cpu2 = hashEngine.hashPreimage('a');
const gpu2 = hashEngine.hashBatchGpu(['a'])[0];
console.log('CPU:', cpu2.substring(0, 64));
console.log('GPU:', gpu2.substring(0, 64));
console.log('Match:', cpu2 === gpu2 ? '✅' : '❌');
console.log();

console.log('Test 3: Empty string');
console.log('-----------------------------------');
const cpu3 = hashEngine.hashPreimage('');
const gpu3 = hashEngine.hashBatchGpu([''])[0];
console.log('CPU:', cpu3.substring(0, 64));
console.log('GPU:', gpu3.substring(0, 64));
console.log('Match:', cpu3 === gpu3 ? '✅' : '❌');
console.log();

console.log('Test 4: Numbers only - "12345"');
console.log('-----------------------------------');
const cpu4 = hashEngine.hashPreimage('12345');
const gpu4 = hashEngine.hashBatchGpu(['12345'])[0];
console.log('CPU:', cpu4.substring(0, 64));
console.log('GPU:', gpu4.substring(0, 64));
console.log('Match:', cpu4 === gpu4 ? '✅' : '❌');
console.log();

console.log('Test 5: Node.js raw Blake2b on "abc" (for reference)');
console.log('-----------------------------------');
const nodeBlake = crypto.createHash('blake2b512').update('abc').digest('hex');
console.log('Node Blake2b:', nodeBlake);
console.log('(This will differ from HashEngine because HashEngine does more than just Blake2b)');
console.log();

console.log('====================================');
console.log('Analysis');
console.log('====================================');
console.log('If ALL tests fail: Core Blake2b implementation is wrong');
console.log('If SOME tests fail: Edge case or data handling bug');
console.log('If empty string works but others fail: Input processing bug');
