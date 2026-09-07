#include <immintrin.h>
#include "convolution.h"

void conv_optimized(const float* in, float* out, const float* ker, int H, int W, int K) {
    const int PW = W + 2;   // K is assumed to be 3

    __m256 w0 = _mm256_set1_ps(ker[0]);
    __m256 w1 = _mm256_set1_ps(ker[1]);
    __m256 w2 = _mm256_set1_ps(ker[2]);
    __m256 w3 = _mm256_set1_ps(ker[3]);
    __m256 w4 = _mm256_set1_ps(ker[4]);
    __m256 w5 = _mm256_set1_ps(ker[5]);
    __m256 w6 = _mm256_set1_ps(ker[6]);
    __m256 w7 = _mm256_set1_ps(ker[7]);
    __m256 w8 = _mm256_set1_ps(ker[8]);

    for (int y = 0; y < H; y++) {
        const float* r0 = in + (y + 0) * PW;
        const float* r1 = in + (y + 1) * PW;
        const float* r2 = in + (y + 2) * PW;
        for (int x = 0; x < W; x += 8) {
            __m256 acc = _mm256_mul_ps(_mm256_loadu_ps(r0 + x),     w0);
            acc = _mm256_fmadd_ps(_mm256_loadu_ps(r0 + x + 1), w1, acc);
            acc = _mm256_fmadd_ps(_mm256_loadu_ps(r0 + x + 2), w2, acc);
            acc = _mm256_fmadd_ps(_mm256_loadu_ps(r1 + x),     w3, acc);
            acc = _mm256_fmadd_ps(_mm256_loadu_ps(r1 + x + 1), w4, acc);
            acc = _mm256_fmadd_ps(_mm256_loadu_ps(r1 + x + 2), w5, acc);
            acc = _mm256_fmadd_ps(_mm256_loadu_ps(r2 + x),     w6, acc);
            acc = _mm256_fmadd_ps(_mm256_loadu_ps(r2 + x + 1), w7, acc);
            acc = _mm256_fmadd_ps(_mm256_loadu_ps(r2 + x + 2), w8, acc);
            _mm256_storeu_ps(out + y * W + x, acc);
        }
    }
}
