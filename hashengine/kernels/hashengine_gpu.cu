/**
 * HashEngine GPU - Complete CUDA implementation
 * Combines Blake2b, Argon2, and VM execution for GPU mining
 *
 * This single-file implementation includes all required kernels for full GPU mining
 */

#include <stdint.h>
#include <stdio.h>

// ============================================================================
// Constants and Configuration
// ============================================================================

#define BLAKE2B_BLOCKBYTES 128
#define BLAKE2B_OUTBYTES 64
#define BLAKE2B_ROUNDS 12

#define NB_REGS 32
#define REGS_BITS 5
#define REGS_INDEX_MASK 0x1F
#define INSTR_SIZE 20
#define REGISTER_SIZE 8

// Instruction opcodes
#define OP_ADD 0
#define OP_MUL 1
#define OP_MULH 2
#define OP_XOR 3
#define OP_DIV 4
#define OP_MOD 5
#define OP_AND 6
#define OP_HASH 7
#define OP_ISQRT 8
#define OP_NEG 9
#define OP_BITREV 10
#define OP_ROTL 11
#define OP_ROTR 12

// Operand types
#define OPERAND_REG 0
#define OPERAND_MEMORY 1
#define OPERAND_LITERAL 2
#define OPERAND_SPECIAL1 3
#define OPERAND_SPECIAL2 4

// ============================================================================
// Blake2b Implementation
// Includes blake2b_512_ref() and blake2b_long_ref() with all bug fixes applied
// Also includes blake2b_ctx_* functions for VM state management
// ============================================================================

// IV constants for Blake2b-512
__constant__ uint64_t blake2b_IV[8] = {
    0x6a09e667f3bcc908ULL, 0xbb67ae8584caa73bULL,
    0x3c6ef372fe94f82bULL, 0xa54ff53a5f1d36f1ULL,
    0x510e527fade682d1ULL, 0x9b05688c2b3e6c1fULL,
    0x1f83d9abfb41bd6bULL, 0x5be0cd19137e2179ULL
};

// Sigma permutation table
__constant__ uint8_t blake2b_sigma[12][16] = {
    {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
    {14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3},
    {11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4},
    {7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8},
    {9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13},
    {2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9},
    {12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11},
    {13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10},
    {6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5},
    {10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0},
    {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
    {14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3}
};

__device__ __forceinline__ uint64_t rotr64(uint64_t x, int n) {
    return (x >> n) | (x << (64 - n));
}

__device__ void blake2b_G(uint64_t *v, int a, int b, int c, int d, uint64_t x, uint64_t y) {
    v[a] = v[a] + v[b] + x;
    v[d] = rotr64(v[d] ^ v[a], 32);
    v[c] = v[c] + v[d];
    v[b] = rotr64(v[b] ^ v[c], 24);
    v[a] = v[a] + v[b] + y;
    v[d] = rotr64(v[d] ^ v[a], 16);
    v[c] = v[c] + v[d];
    v[b] = rotr64(v[b] ^ v[c], 63);
}

__device__ void blake2b_compress(uint64_t h[8], const uint64_t m[16], uint64_t t, uint64_t f) {
    uint64_t v[16];

    for (int i = 0; i < 8; i++) {
        v[i] = h[i];
        v[i + 8] = blake2b_IV[i];
    }

    v[12] ^= t;
    v[13] ^= 0;
    v[14] ^= f;

    for (int round = 0; round < BLAKE2B_ROUNDS; round++) {
        blake2b_G(v, 0, 4, 8, 12, m[blake2b_sigma[round][0]], m[blake2b_sigma[round][1]]);
        blake2b_G(v, 1, 5, 9, 13, m[blake2b_sigma[round][2]], m[blake2b_sigma[round][3]]);
        blake2b_G(v, 2, 6, 10, 14, m[blake2b_sigma[round][4]], m[blake2b_sigma[round][5]]);
        blake2b_G(v, 3, 7, 11, 15, m[blake2b_sigma[round][6]], m[blake2b_sigma[round][7]]);

        blake2b_G(v, 0, 5, 10, 15, m[blake2b_sigma[round][8]], m[blake2b_sigma[round][9]]);
        blake2b_G(v, 1, 6, 11, 12, m[blake2b_sigma[round][10]], m[blake2b_sigma[round][11]]);
        blake2b_G(v, 2, 7, 8, 13, m[blake2b_sigma[round][12]], m[blake2b_sigma[round][13]]);
        blake2b_G(v, 3, 4, 9, 14, m[blake2b_sigma[round][14]], m[blake2b_sigma[round][15]]);
    }

    for (int i = 0; i < 8; i++) {
        h[i] ^= v[i] ^ v[i + 8];
    }
}

// Blake2b-512 implementation (this was always correct)
__device__ void blake2b_512_ref(const uint8_t *input, size_t input_len, uint8_t *output) {
    uint64_t h[8];
    uint8_t block[BLAKE2B_BLOCKBYTES];
    uint64_t m[16];

    for (int i = 0; i < 8; i++) {
        h[i] = blake2b_IV[i];
    }
    h[0] ^= 0x01010000 ^ BLAKE2B_OUTBYTES;

    size_t offset = 0;

    while (input_len > BLAKE2B_BLOCKBYTES) {
        for (int i = 0; i < BLAKE2B_BLOCKBYTES; i++) {
            block[i] = input[offset + i];
        }

        for (int i = 0; i < 16; i++) {
            m[i] = 0;
            for (int j = 0; j < 8; j++) {
                m[i] |= ((uint64_t)block[i * 8 + j]) << (j * 8);
            }
        }

        offset += BLAKE2B_BLOCKBYTES;
        blake2b_compress(h, m, offset, 0);
        input_len -= BLAKE2B_BLOCKBYTES;
    }

    for (int i = 0; i < BLAKE2B_BLOCKBYTES; i++) {
        block[i] = 0;
    }
    for (size_t i = 0; i < input_len; i++) {
        block[i] = input[offset + i];
    }

    for (int i = 0; i < 16; i++) {
        m[i] = 0;
        for (int j = 0; j < 8; j++) {
            m[i] |= ((uint64_t)block[i * 8 + j]) << (j * 8);
        }
    }

    offset += input_len;
    blake2b_compress(h, m, offset, 0xFFFFFFFFFFFFFFFFULL);

    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            output[i * 8 + j] = (h[i] >> (j * 8)) & 0xFF;
        }
    }
}

// ============================================================================
// Argon2 hprime Implementation
// ============================================================================

__device__ void blake2b_long_ref(const uint8_t *input, size_t input_len, uint8_t *output, size_t output_len) {
    uint8_t out_buffer[64];

    if (output_len <= 64) {
        blake2b_512_ref(input, input_len, output);
        return;
    }

    uint32_t toproduce = output_len;
    uint8_t out_len_bytes[4];
    out_len_bytes[0] = output_len & 0xFF;
    out_len_bytes[1] = (output_len >> 8) & 0xFF;
    out_len_bytes[2] = (output_len >> 16) & 0xFF;
    out_len_bytes[3] = (output_len >> 24) & 0xFF;

    uint8_t first_input[512];
    for (int i = 0; i < 4; i++) {
        first_input[i] = out_len_bytes[i];
    }
    size_t copy_len = (input_len > 508) ? 508 : input_len;
    for (size_t i = 0; i < copy_len; i++) {
        first_input[4 + i] = input[i];
    }

    blake2b_512_ref(first_input, 4 + copy_len, out_buffer);

    uint32_t out_pos = 0;
    uint32_t to_copy = (toproduce > 64) ? 64 : toproduce;

    for (uint32_t i = 0; i < to_copy; i++) {
        output[out_pos++] = out_buffer[i];
    }
    toproduce -= to_copy;

    while (toproduce > 0) {
        blake2b_512_ref(out_buffer, 64, out_buffer);
        to_copy = (toproduce > 64) ? 64 : toproduce;
        for (uint32_t i = 0; i < to_copy; i++) {
            output[out_pos++] = out_buffer[i];
        }
        toproduce -= to_copy;
    }
}

// End of Blake2b implementation (bugs fixed: line 206 and line 209)

// ============================================================================
// VM State and Operations
// ============================================================================

struct Blake2bContext {
    uint64_t h[8];
    uint8_t buffer[BLAKE2B_BLOCKBYTES];
    size_t buffer_len;
    size_t total_len;
};

__device__ void blake2b_ctx_init(Blake2bContext *ctx) {
    for (int i = 0; i < 8; i++) {
        ctx->h[i] = blake2b_IV[i];
    }
    ctx->h[0] ^= 0x01010000 ^ BLAKE2B_OUTBYTES;
    ctx->buffer_len = 0;
    ctx->total_len = 0;
}

__device__ void blake2b_ctx_update(Blake2bContext *ctx, const uint8_t *data, size_t len) {
    ctx->total_len += len;

    if (ctx->buffer_len + len <= BLAKE2B_BLOCKBYTES) {
        for (size_t i = 0; i < len; i++) {
            ctx->buffer[ctx->buffer_len + i] = data[i];
        }
        ctx->buffer_len += len;
        return;
    }

    size_t data_offset = 0;
    if (ctx->buffer_len > 0) {
        size_t to_copy = BLAKE2B_BLOCKBYTES - ctx->buffer_len;
        for (size_t i = 0; i < to_copy; i++) {
            ctx->buffer[ctx->buffer_len + i] = data[i];
        }

        uint64_t m[16];
        for (int i = 0; i < 16; i++) {
            m[i] = 0;
            for (int j = 0; j < 8; j++) {
                m[i] |= ((uint64_t)ctx->buffer[i * 8 + j]) << (j * 8);
            }
        }
        blake2b_compress(ctx->h, m, ctx->total_len - len + to_copy, 0);

        data_offset = to_copy;
        len -= to_copy;
        ctx->buffer_len = 0;
    }

    while (len > BLAKE2B_BLOCKBYTES) {
        uint64_t m[16];
        for (int i = 0; i < 16; i++) {
            m[i] = 0;
            for (int j = 0; j < 8; j++) {
                m[i] |= ((uint64_t)data[data_offset + i * 8 + j]) << (j * 8);
            }
        }
        data_offset += BLAKE2B_BLOCKBYTES;
        blake2b_compress(ctx->h, m, ctx->total_len - len + BLAKE2B_BLOCKBYTES, 0);
        len -= BLAKE2B_BLOCKBYTES;
    }

    if (len > 0) {
        for (size_t i = 0; i < len; i++) {
            ctx->buffer[i] = data[data_offset + i];
        }
        ctx->buffer_len = len;
    }
}

__device__ void blake2b_ctx_finalize(Blake2bContext *ctx, uint8_t *output) {
    uint8_t block[BLAKE2B_BLOCKBYTES];
    for (int i = 0; i < BLAKE2B_BLOCKBYTES; i++) {
        block[i] = 0;
    }
    for (size_t i = 0; i < ctx->buffer_len; i++) {
        block[i] = ctx->buffer[i];
    }

    uint64_t m[16];
    for (int i = 0; i < 16; i++) {
        m[i] = 0;
        for (int j = 0; j < 8; j++) {
            m[i] |= ((uint64_t)block[i * 8 + j]) << (j * 8);
        }
    }

    blake2b_compress(ctx->h, m, ctx->total_len, 0xFFFFFFFFFFFFFFFFULL);

    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            output[i * 8 + j] = (ctx->h[i] >> (j * 8)) & 0xFF;
        }
    }
}

struct VMState {
    uint64_t regs[NB_REGS];
    Blake2bContext prog_digest;
    Blake2bContext mem_digest;
    uint8_t prog_seed[64];
    uint32_t ip;
    uint32_t memory_counter;
    uint32_t loop_counter;
};

__device__ __forceinline__ uint8_t decode_opcode(uint8_t op_byte) {
    if (op_byte < 40) return OP_ADD;
    if (op_byte < 80) return OP_MUL;
    if (op_byte < 96) return OP_MULH;
    if (op_byte < 112) return OP_DIV;
    if (op_byte < 128) return OP_MOD;
    if (op_byte < 138) return OP_ISQRT;
    if (op_byte < 148) return OP_BITREV;
    if (op_byte < 188) return OP_XOR;
    if (op_byte < 204) return OP_ROTL;
    if (op_byte < 220) return OP_ROTR;
    if (op_byte < 240) return OP_NEG;
    if (op_byte < 248) return OP_AND;
    return OP_HASH;
}

__device__ __forceinline__ uint8_t decode_operand(uint8_t op_nibble) {
    if (op_nibble < 5) return OPERAND_REG;
    if (op_nibble < 9) return OPERAND_MEMORY;
    if (op_nibble < 13) return OPERAND_LITERAL;
    if (op_nibble < 14) return OPERAND_SPECIAL1;
    return OPERAND_SPECIAL2;
}

__device__ __forceinline__ uint64_t isqrt64(uint64_t n) {
    if (n == 0) return 0;
    uint64_t x = n;
    uint64_t y = (x + 1) / 2;
    while (y < x) {
        x = y;
        y = (x + n / x) / 2;
    }
    return x;
}

__device__ uint64_t rom_access(const uint8_t *rom, uint32_t addr, uint32_t rom_size, VMState *vm) {
    uint32_t rom_addr = (addr % rom_size) & ~63UL;
    const uint8_t *mem = rom + rom_addr;

    blake2b_ctx_update(&vm->mem_digest, mem, 64);
    vm->memory_counter++;

    uint32_t idx = ((vm->memory_counter % 8) * 8);
    uint64_t val = 0;
    for (int i = 0; i < 8; i++) {
        val |= ((uint64_t)mem[idx + i]) << (i * 8);
    }
    return val;
}

__device__ uint64_t special1_value(VMState *vm) {
    Blake2bContext ctx_copy = vm->prog_digest;
    uint8_t hash[64];
    blake2b_ctx_finalize(&ctx_copy, hash);
    uint64_t val = 0;
    for (int i = 0; i < 8; i++) {
        val |= ((uint64_t)hash[i]) << (i * 8);
    }
    return val;
}

__device__ uint64_t special2_value(VMState *vm) {
    Blake2bContext ctx_copy = vm->mem_digest;
    uint8_t hash[64];
    blake2b_ctx_finalize(&ctx_copy, hash);
    uint64_t val = 0;
    for (int i = 0; i < 8; i++) {
        val |= ((uint64_t)hash[i]) << (i * 8);
    }
    return val;
}

__device__ void execute_instruction(
    VMState *vm,
    const uint8_t *program,
    const uint8_t *rom,
    uint32_t rom_size,
    uint32_t nb_instrs
) {
    uint32_t instr_addr = (vm->ip * INSTR_SIZE) % (nb_instrs * INSTR_SIZE);
    const uint8_t *instr = program + instr_addr;

    uint8_t opcode = decode_opcode(instr[0]);
    uint8_t op1_type = decode_operand(instr[1] >> 4);
    uint8_t op2_type = decode_operand(instr[1] & 0x0F);

    uint16_t rs = ((uint16_t)instr[2] << 8) | (uint16_t)instr[3];
    uint8_t r1 = ((rs >> (2 * REGS_BITS)) & REGS_INDEX_MASK);
    uint8_t r2 = ((rs >> REGS_BITS) & REGS_INDEX_MASK);
    uint8_t r3 = (rs & REGS_INDEX_MASK);

    uint64_t lit1 = 0, lit2 = 0;
    for (int i = 0; i < 8; i++) {
        lit1 |= ((uint64_t)instr[4 + i]) << (i * 8);
        lit2 |= ((uint64_t)instr[12 + i]) << (i * 8);
    }

    uint64_t src1 = 0, src2 = 0;

    switch (op1_type) {
        case OPERAND_REG: src1 = vm->regs[r1]; break;
        case OPERAND_MEMORY: src1 = rom_access(rom, lit1, rom_size, vm); break;
        case OPERAND_LITERAL: src1 = lit1; break;
        case OPERAND_SPECIAL1: src1 = special1_value(vm); break;
        case OPERAND_SPECIAL2: src1 = special2_value(vm); break;
    }

    switch (op2_type) {
        case OPERAND_REG: src2 = vm->regs[r2]; break;
        case OPERAND_MEMORY: src2 = rom_access(rom, lit2, rom_size, vm); break;
        case OPERAND_LITERAL: src2 = lit2; break;
        case OPERAND_SPECIAL1: src2 = special1_value(vm); break;
        case OPERAND_SPECIAL2: src2 = special2_value(vm); break;
    }

    uint64_t result = 0;

    switch (opcode) {
        case OP_ADD:
            result = src1 + src2;
            break;
        case OP_MUL:
            result = src1 * src2;
            break;
        case OP_MULH: {
            unsigned long long high;
            __umul64hi(src1, src2);
            result = __umul64hi(src1, src2);
            break;
        }
        case OP_XOR:
            result = src1 ^ src2;
            break;
        case OP_DIV:
            result = (src2 == 0) ? special1_value(vm) : (src1 / src2);
            break;
        case OP_MOD:
            result = (src2 == 0) ? special1_value(vm) : (src1 % src2);
            break;
        case OP_AND:
            result = src1 & src2;
            break;
        case OP_HASH: {
            uint8_t hash_input[16];
            for (int i = 0; i < 8; i++) {
                hash_input[i] = (src1 >> (i * 8)) & 0xFF;
                hash_input[8 + i] = (src2 >> (i * 8)) & 0xFF;
            }
            uint8_t hash_out[64];
            blake2b_512_ref(hash_input, 16, hash_out);

            uint8_t hash_variant = instr[0] - 248;
            result = 0;
            for (int i = 0; i < 8; i++) {
                result |= ((uint64_t)hash_out[hash_variant * 8 + i]) << (i * 8);
            }
            break;
        }
        case OP_ISQRT:
            result = isqrt64(src1);
            break;
        case OP_NEG:
            result = ~src1;
            break;
        case OP_BITREV:
            result = __brevll(src1);
            break;
        case OP_ROTL:
            result = (src1 << (r1 & 63)) | (src1 >> (64 - (r1 & 63)));
            break;
        case OP_ROTR:
            result = (src1 >> (r1 & 63)) | (src1 << (64 - (r1 & 63)));
            break;
    }

    vm->regs[r3] = result;
    blake2b_ctx_update(&vm->prog_digest, instr, INSTR_SIZE);
    vm->ip++;
}

// ============================================================================
// Main GPU Mining Kernel
// ============================================================================

extern "C" __global__ void hashengine_mine_batch(
    const uint8_t *preimages_data,     // Concatenated preimage strings
    const uint32_t *preimage_offsets,  // Offset for each preimage
    const uint32_t *preimage_lengths,  // Length of each preimage
    const uint8_t *rom_data,           // ROM data (64 bytes)
    const uint8_t *rom_digest,         // ROM digest
    uint32_t rom_size,
    uint32_t nb_loops,
    uint32_t nb_instrs,
    uint8_t *output_hashes,            // Output: 64 bytes per hash
    uint32_t batch_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;

    // Get preimage for this thread
    uint32_t preimage_offset = preimage_offsets[idx];
    uint32_t preimage_len = preimage_lengths[idx];
    const uint8_t *preimage = preimages_data + preimage_offset;

    // Initialize VM using Argon2 hprime
    const size_t DIGEST_INIT_SIZE = 64;
    const size_t REGS_CONTENT_SIZE = REGISTER_SIZE * NB_REGS;
    const size_t INIT_BUFFER_SIZE = REGS_CONTENT_SIZE + 3 * DIGEST_INIT_SIZE;

    uint8_t init_buffer[INIT_BUFFER_SIZE];

    // Prepare input: rom_digest || preimage
    uint8_t hprime_input[512];
    for (int i = 0; i < 64; i++) {
        hprime_input[i] = rom_digest[i];
    }
    uint32_t actual_preimage_len = (preimage_len < 448) ? preimage_len : 448;
    for (uint32_t i = 0; i < actual_preimage_len; i++) {
        hprime_input[64 + i] = preimage[i];
    }

    // Run Argon2 hprime to initialize VM state
    blake2b_long_ref(hprime_input, 64 + actual_preimage_len, init_buffer, INIT_BUFFER_SIZE);

    // Initialize VM state
    VMState vm;

    // Load registers from init_buffer
    for (int i = 0; i < NB_REGS; i++) {
        vm.regs[i] = 0;
        for (int j = 0; j < 8; j++) {
            vm.regs[i] |= ((uint64_t)init_buffer[i * 8 + j]) << (j * 8);
        }
    }

    // Initialize Blake2b contexts
    blake2b_ctx_init(&vm.prog_digest);
    blake2b_ctx_init(&vm.mem_digest);

    // Update contexts with initialization data
    blake2b_ctx_update(&vm.prog_digest, init_buffer + REGS_CONTENT_SIZE, DIGEST_INIT_SIZE);
    blake2b_ctx_update(&vm.mem_digest, init_buffer + REGS_CONTENT_SIZE + DIGEST_INIT_SIZE, DIGEST_INIT_SIZE);

    // Set prog_seed
    for (int i = 0; i < 64; i++) {
        vm.prog_seed[i] = init_buffer[REGS_CONTENT_SIZE + 2 * DIGEST_INIT_SIZE + i];
    }

    vm.ip = 0;
    vm.memory_counter = 0;
    vm.loop_counter = 0;

    // Prepare program (shuffled by prog_seed)
    uint8_t program[256 * INSTR_SIZE];  // nb_instrs=256 max
    blake2b_long_ref(vm.prog_seed, 64, program, nb_instrs * INSTR_SIZE);

    // Execute VM loops
    for (uint32_t loop = 0; loop < nb_loops; loop++) {
        // Execute all instructions
        for (uint32_t i = 0; i < nb_instrs; i++) {
            execute_instruction(&vm, program, rom_data, rom_size, nb_instrs);
        }

        // Post-instructions processing
        uint64_t sum_regs = 0;
        for (int i = 0; i < NB_REGS; i++) {
            sum_regs += vm.regs[i];
        }

        uint8_t sum_bytes[8];
        for (int i = 0; i < 8; i++) {
            sum_bytes[i] = (sum_regs >> (i * 8)) & 0xFF;
        }

        // Finalize prog and mem digests
        Blake2bContext prog_ctx_copy = vm.prog_digest;
        Blake2bContext mem_ctx_copy = vm.mem_digest;
        blake2b_ctx_update(&prog_ctx_copy, sum_bytes, 8);
        blake2b_ctx_update(&mem_ctx_copy, sum_bytes, 8);

        uint8_t prog_value[64];
        uint8_t mem_value[64];
        blake2b_ctx_finalize(&prog_ctx_copy, prog_value);
        blake2b_ctx_finalize(&mem_ctx_copy, mem_value);

        // Create mixing value
        uint8_t loop_bytes[4];
        for (int i = 0; i < 4; i++) {
            loop_bytes[i] = (vm.loop_counter >> (i * 8)) & 0xFF;
        }

        uint8_t mixing_input[128 + 4];
        for (int i = 0; i < 64; i++) {
            mixing_input[i] = prog_value[i];
            mixing_input[64 + i] = mem_value[i];
        }
        for (int i = 0; i < 4; i++) {
            mixing_input[128 + i] = loop_bytes[i];
        }

        uint8_t mixing_value[64];
        blake2b_512_ref(mixing_input, 132, mixing_value);

        // Generate mixing output
        uint8_t mixing_out[NB_REGS * REGISTER_SIZE * 32];
        blake2b_long_ref(mixing_value, 64, mixing_out, NB_REGS * REGISTER_SIZE * 32);

        // XOR registers with mixing output
        for (int chunk = 0; chunk < 32; chunk++) {
            for (int i = 0; i < NB_REGS; i++) {
                uint64_t mix_val = 0;
                int offset = chunk * NB_REGS * REGISTER_SIZE + i * REGISTER_SIZE;
                for (int j = 0; j < 8; j++) {
                    mix_val |= ((uint64_t)mixing_out[offset + j]) << (j * 8);
                }
                vm.regs[i] ^= mix_val;
            }
        }

        // Update prog_seed and loop_counter
        for (int i = 0; i < 64; i++) {
            vm.prog_seed[i] = prog_value[i];
        }
        vm.loop_counter++;

        // Reshuffle program for next loop
        if (loop + 1 < nb_loops) {
            blake2b_long_ref(vm.prog_seed, 64, program, nb_instrs * INSTR_SIZE);
        }
    }

    // Finalize hash
    uint8_t prog_digest_final[64];
    uint8_t mem_digest_final[64];
    blake2b_ctx_finalize(&vm.prog_digest, prog_digest_final);
    blake2b_ctx_finalize(&vm.mem_digest, mem_digest_final);

    // Final hash = Blake2b(prog_digest || mem_digest || memory_counter || registers)
    uint8_t final_input[64 + 64 + 4 + NB_REGS * 8];
    int pos = 0;

    for (int i = 0; i < 64; i++) {
        final_input[pos++] = prog_digest_final[i];
    }
    for (int i = 0; i < 64; i++) {
        final_input[pos++] = mem_digest_final[i];
    }
    for (int i = 0; i < 4; i++) {
        final_input[pos++] = (vm.memory_counter >> (i * 8)) & 0xFF;
    }
    for (int i = 0; i < NB_REGS; i++) {
        for (int j = 0; j < 8; j++) {
            final_input[pos++] = (vm.regs[i] >> (j * 8)) & 0xFF;
        }
    }

    blake2b_512_ref(final_input, pos, output_hashes + idx * 64);
}
