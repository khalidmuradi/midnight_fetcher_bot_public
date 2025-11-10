/**
 * Test Blake2b-512 against known test vectors
 * https://github.com/BLAKE2/BLAKE2/blob/master/testvectors/blake2b-kat.txt
 */

const crypto = require('crypto');

// Official Blake2b-512 test vectors
const vectors = [
  {
    input: Buffer.from('', 'utf8'),
    expected: '786a02f742015903c6c6fd852552d272912f4740e15847618a86e217f71f5419d25e1031afee585313896444934eb04b903a685b1448b755d56f701afe9be2ce'
  },
  {
    input: Buffer.from('abc', 'utf8'),
    expected: 'ba80a53f981c4d0d6a2797b69f12f6e94c212f14685ac4b74b12bb6fdbffa2d17d87c5392aab792dc252d5de4533cc9518d38aa8dbf1925ab92386edd4009923'
  },
  {
    input: Buffer.from('The quick brown fox jumps over the lazy dog', 'utf8'),
    expected: 'a8add4bdddfd93e4877d2746e62817b116364a1fa7bc148d95090bc7333b3673f82401cf7aa2e4cb1ecd90296e3f14cb5413f8ed77be73045b13914cdcd6a918'
  }
];

console.log('Testing Node.js crypto Blake2b-512 implementation:');
console.log();

let pass = 0;
let fail = 0;

for (let i = 0; i < vectors.length; i++) {
  const v = vectors[i];
  const hash = crypto.createHash('blake2b512').update(v.input).digest('hex');
  const match = hash === v.expected;

  const inputStr = v.input.toString('utf8') || '(empty)';
  console.log(`Test ${i + 1}: "${inputStr}"`);
  console.log(`  Expected: ${v.expected.substring(0, 40)}...`);
  console.log(`  Got:      ${hash.substring(0, 40)}...`);
  console.log(`  Result:   ${match ? '✅ PASS' : '❌ FAIL'}`);
  console.log();

  if (match) pass++;
  else fail++;
}

console.log('========================================');
console.log(`Results: ${pass}/${vectors.length} passed`);
console.log();

if (pass === vectors.length) {
  console.log('✅ Node.js Blake2b-512 implementation is correct');
} else {
  console.log('❌ Node.js Blake2b-512 has issues (unexpected!)');
}
