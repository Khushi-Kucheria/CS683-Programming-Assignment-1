// conv_simd.cpp
#include "convolution.h"
#include <immintrin.h>

void conv_simd(const float* in, float* out, const float* ker,
               int H, int W, int K) {
    const int p = K / 2;
    const int in_stride = W + 2 * p;

#if defined(WIDTH_512)
    const int VEC = 16;
#elif defined(WIDTH_256)
    const int VEC = 8;
#else
    const int VEC = 4;   // default: 128-bit
#endif

    for (int oy = 0; oy < H; ++oy) {
        int ox = 0;

        // vectorized main loop: process VEC output pixels at once
        for (; ox + VEC <= W; ox += VEC) {

#if defined(WIDTH_512)
            __m512 acc = _mm512_setzero_ps();
#elif defined(WIDTH_256)
            __m256 acc = _mm256_setzero_ps();
#else
            __m128 acc = _mm_setzero_ps();
#endif

            for (int ky = 0; ky < K; ++ky) {
                for (int kx = 0; kx < K; ++kx) {
                    float w = ker[ky * K + kx];
                    const float* row_ptr = &in[(oy + ky) * in_stride + (ox + kx)];

#if defined(WIDTH_512)
                    __m512 in_vec  = _mm512_loadu_ps(row_ptr);
                    __m512 ker_vec = _mm512_set1_ps(w);
                    acc = _mm512_fmadd_ps(in_vec, ker_vec, acc);
#elif defined(WIDTH_256)
                    __m256 in_vec  = _mm256_loadu_ps(row_ptr);
                    __m256 ker_vec = _mm256_set1_ps(w);
                    acc = _mm256_fmadd_ps(in_vec, ker_vec, acc);
#else
                    __m128 in_vec  = _mm_loadu_ps(row_ptr);
                    __m128 ker_vec = _mm_set1_ps(w);
                    acc = _mm_add_ps(acc, _mm_mul_ps(in_vec, ker_vec));
#endif
                }
            }

#if defined(WIDTH_512)
            _mm512_storeu_ps(&out[oy * W + ox], acc);
#elif defined(WIDTH_256)
            _mm256_storeu_ps(&out[oy * W + ox], acc);
#else
            _mm_storeu_ps(&out[oy * W + ox], acc);
#endif
        }

        // remainder: leftover output pixels that don't fill a full vector
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
