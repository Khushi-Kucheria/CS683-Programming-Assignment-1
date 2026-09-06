// conv_simd.cpp  STAGE 4: SIMD with AVX2 intrinsics
#include <immintrin.h>
#include "convolution.h"

void conv_simd(const float* in, float* out, const float* ker,
               int H, int W, int K) {
    const int p = K / 2;
    const int in_stride = W + 2 * p;

    for (int oy = 0; oy < H; ++oy) {
        int ox = 0;

        // Vectorized main loop: 8 output pixels at a time (AVX2 = 256-bit = 8 floats)
        for (; ox + 8 <= W; ox += 8) {
            __m256 acc = _mm256_setzero_ps();

            for (int ky = 0; ky < K; ++ky) {
                for (int kx = 0; kx < K; ++kx) {
                    float w = ker[ky * K + kx];
                    const float* row_ptr = &in[(oy + ky) * in_stride + (ox + kx)];

                    __m256 in_vec  = _mm256_loadu_ps(row_ptr);
                    __m256 ker_vec = _mm256_set1_ps(w);
                    acc = _mm256_fmadd_ps(in_vec, ker_vec, acc);
                }
            }

            _mm256_storeu_ps(&out[oy * W + ox], acc);
        }

        // Remainder: leftover pixels in this row that don't fill a full 8-wide vector
        for (; ox < W; ++ox) {
            float acc = 0.0f;
            for (int ky = 0; ky < K; ++ky) {
                for (int kx = 0; kx < K; ++kx) {
                    acc += in[(oy + ky) * in_stride + (ox + kx)] * ker[ky * K + kx];
                }
            }
            out[oy * W + ox] = acc;
        }
    }
}
