/**
 * Count leading zero bits in hash
 */

const hash = '00000624729811a70d23d54ff1a0f47af21e5689d2d06902aca34159fe796d2ec92cbc11983679bcbcf606c8d4c1c9fd757c024ef51db08c4af451e2af986d39';

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

const zeroBits = countLeadingZeroBits(hash);

console.log('Hash:', hash);
console.log('Leading zero bits:', zeroBits);
console.log('Required:', 21);
console.log('Passes difficulty?', zeroBits >= 21 ? 'YES ✅' : 'NO ❌');
console.log();

if (zeroBits < 21) {
  console.log('This hash does NOT meet difficulty requirements.');
  console.log('The GPU is producing INCORRECT hashes.');
  console.log();
  console.log('Binary breakdown of first bytes:');
  for (let i = 0; i < 8; i++) {
    const byte = parseInt(hash.substring(i*2, i*2+2), 16);
    const binary = byte.toString(2).padStart(8, '0');
    console.log(`Byte ${i}: 0x${hash.substring(i*2, i*2+2)} = ${binary}`);
    if (byte !== 0) break;
  }
}
