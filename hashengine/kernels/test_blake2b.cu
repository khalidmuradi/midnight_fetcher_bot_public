/**
 * Standalone Blake2b test to verify correctness
 * Compile with: nvcc test_blake2b.cu -o test_blake2b
 * Run with: ./test_blake2b
 */

#include <stdio.h>
#include <stdint.h>
#include <string.h>

#define BLAKE2B_BLOCKBYTES 128
#define BLAKE2B_OUTBYTES 64

// Include the reference implementation
#include "blake2b_ref.cu"

// Test function to run on GPU
__global__ void test_blake2b_kernel(const char* input, int input_len, uint8_t* output) {
    blake2b_512_ref((const uint8_t*)input, input_len, output);
}

int main() {
    // Test vector: Blake2b-512("abc")
    // Expected: ba80a53f981c4d0d6a2797b69f12f6e94c212f14685ac4b74b12bb6fdbffa2d17d87c5392aab792dc252d5de4533cc9518d38aa8dbf1925ab92386edd4009923

    const char* test_input = "abc";
    int input_len = 3;

    // Allocate device memory
    char* d_input;
    uint8_t* d_output;
    cudaMalloc(&d_input, input_len);
    cudaMalloc(&d_output, 64);

    // Copy input to device
    cudaMemcpy(d_input, test_input, input_len, cudaMemcpyHostToDevice);

    // Run kernel
    test_blake2b_kernel<<<1, 1>>>(d_input, input_len, d_output);
    cudaDeviceSynchronize();

    // Copy result back
    uint8_t result[64];
    cudaMemcpy(result, d_output, 64, cudaMemcpyDeviceToHost);

    // Print result
    printf("Blake2b-512(\"abc\") =\n");
    for (int i = 0; i < 64; i++) {
        printf("%02x", result[i]);
    }
    printf("\n\nExpected:\nba80a53f981c4d0d6a2797b69f12f6e94c212f14685ac4b74b12bb6fdbffa2d17d87c5392aab792dc252d5de4533cc9518d38aa8dbf1925ab92386edd4009923\n");

    // Cleanup
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}
