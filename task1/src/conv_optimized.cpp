// conv_optimized.cpp  STAGE 5: PUT IT ALL TOGETHER
// Hint: measure after every change. Not every "optimization" helps  let the numbers,
// not intuition, decide.

#include <immintrin.h>
#include <algorithm>
#include "convolution.h"
#ifndef TILE
#define TILE 32     
#endif
static void conv_tiled_simd_k3(const float* __restrict__ in, float* __restrict__ out,
                                int H, int W, const float* __restrict__ k) {
    const int OH = H - 3 + 1;
    const int OW = W - 3 + 1;
    __m256 w[9];
    for (int t = 0; t < 9; t++) w[t] = _mm256_set1_ps(k[t]);
    for (int ti = 0; ti < OH; ti += TILE) {
        int imax = std::min(ti + TILE, OH);
        for (int tj = 0; tj < OW; tj += TILE) {
            int jmax = std::min(tj + TILE, OW);
            for (int i = ti; i < imax; i++) {
                const float* r0 = in + (i + 0) * W;
                const float* r1 = in + (i + 1) * W;
                const float* r2 = in + (i + 2) * W;
                int j = tj;
                for (; j <= jmax - 8; j += 8) {
                    __m256 acc = _mm256_mul_ps(_mm256_loadu_ps(r0 + j),     w[0]);
                    acc = _mm256_fmadd_ps(_mm256_loadu_ps(r0 + j + 1), w[1], acc);
                    acc = _mm256_fmadd_ps(_mm256_loadu_ps(r0 + j + 2), w[2], acc);
                    acc = _mm256_fmadd_ps(_mm256_loadu_ps(r1 + j),     w[3], acc);
                    acc = _mm256_fmadd_ps(_mm256_loadu_ps(r1 + j + 1), w[4], acc);
                    acc = _mm256_fmadd_ps(_mm256_loadu_ps(r1 + j + 2), w[5], acc);
                    acc = _mm256_fmadd_ps(_mm256_loadu_ps(r2 + j),     w[6], acc);
                    acc = _mm256_fmadd_ps(_mm256_loadu_ps(r2 + j + 1), w[7], acc);
                    acc = _mm256_fmadd_ps(_mm256_loadu_ps(r2 + j + 2), w[8], acc);
                    _mm256_storeu_ps(out + i * OW + j, acc);
                }
                for (; j < jmax; j++) {   
                    out[i * OW + j] =
                        r0[j]*k[0] + r0[j+1]*k[1] + r0[j+2]*k[2] +
                        r1[j]*k[3] + r1[j+1]*k[4] + r1[j+2]*k[5] +
                        r2[j]*k[6] + r2[j+1]*k[7] + r2[j+2]*k[8];
                }
            }
        }
    }
}
static void conv_tiled_simd_generic(const float* __restrict__ in, float* __restrict__ out,
                                     int H, int W, const float* __restrict__ kernel, int K) {
    const int OH = H - K + 1;
    const int OW = W - K + 1;
    for (int ti = 0; ti < OH; ti += TILE) {
        int imax = std::min(ti + TILE, OH);
        for (int tj = 0; tj < OW; tj += TILE) {
            int jmax = std::min(tj + TILE, OW);
            for (int i = ti; i < imax; i++) {
                int j = tj;
                for (; j <= jmax - 8; j += 8) {
                    __m256 acc = _mm256_setzero_ps();
                    for (int ki = 0; ki < K; ki++) {
                        const float* row = in + (i + ki) * W + j;
                        for (int kj = 0; kj < K; kj++) {
                            __m256 wv = _mm256_set1_ps(kernel[ki * K + kj]);
                            acc = _mm256_fmadd_ps(_mm256_loadu_ps(row + kj), wv, acc);
                        }
                    }
                    _mm256_storeu_ps(out + i * OW + j, acc);
                }
                for (; j < jmax; j++) {
                    float acc = 0.0f;
                    for (int ki = 0; ki < K; ki++) {
                        const float* row = in + (i + ki) * W + j;
                        for (int kj = 0; kj < K; kj++)
                            acc += row[kj] * kernel[ki * K + kj];
                    }
                    out[i * OW + j] = acc;
                }
            }
        }
    }
}
void conv_tiled_simd(const float* input, float* output,
                      int H, int W, const float* kernel, int K) {
    if (K == 3) conv_tiled_simd_k3(input, output, H, W, kernel);
    else        conv_tiled_simd_generic(input, output, H, W, kernel, K);
}


