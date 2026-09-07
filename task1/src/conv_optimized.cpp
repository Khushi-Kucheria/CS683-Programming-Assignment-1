#include <immintrin.h>
#include <algorithm>
#include "convolution.h"

#ifndef TILE
#define TILE 32
#endif
static_assert(TILE % 8 == 0, "TILE must be a multiple of 8 to keep every tile 8-aligned");

static void conv_optimized_k3(const float* __restrict__ in, float* __restrict__ out,
                               const float* __restrict__ ker, int H, int W) {
    const int PW = W + 2;

    __m256 w[9];
    for (int t = 0; t < 9; t++) w[t] = _mm256_set1_ps(ker[t]);

    for (int ty = 0; ty < H; ty += TILE) {
        int ymax = std::min(ty + TILE, H);
        for (int tx = 0; tx < W; tx += TILE) {
            int xmax = std::min(tx + TILE, W);
            for (int y = ty; y < ymax; y++) {
                const float* r0 = in + (y + 0) * PW;
                const float* r1 = in + (y + 1) * PW;
                const float* r2 = in + (y + 2) * PW;
                for (int x = tx; x < xmax; x += 8) {
                    __m256 acc = _mm256_mul_ps(_mm256_loadu_ps(r0 + x),     w[0]);
                    acc = _mm256_fmadd_ps(_mm256_loadu_ps(r0 + x + 1), w[1], acc);
                    acc = _mm256_fmadd_ps(_mm256_loadu_ps(r0 + x + 2), w[2], acc);
                    acc = _mm256_fmadd_ps(_mm256_loadu_ps(r1 + x),     w[3], acc);
                    acc = _mm256_fmadd_ps(_mm256_loadu_ps(r1 + x + 1), w[4], acc);
                    acc = _mm256_fmadd_ps(_mm256_loadu_ps(r1 + x + 2), w[5], acc);
                    acc = _mm256_fmadd_ps(_mm256_loadu_ps(r2 + x),     w[6], acc);
                    acc = _mm256_fmadd_ps(_mm256_loadu_ps(r2 + x + 1), w[7], acc);
                    acc = _mm256_fmadd_ps(_mm256_loadu_ps(r2 + x + 2), w[8], acc);
                    _mm256_storeu_ps(out + y * W + x, acc);
                }
            }
        }
    }
}

static void conv_optimized_generic(const float* __restrict__ in, float* __restrict__ out,
                                    const float* __restrict__ ker, int H, int W, int K) {
    const int p = K / 2;
    const int PW = W + 2 * p;

    for (int ty = 0; ty < H; ty += TILE) {
        int ymax = std::min(ty + TILE, H);
        for (int tx = 0; tx < W; tx += TILE) {
            int xmax = std::min(tx + TILE, W);
            for (int y = ty; y < ymax; y++) {
                for (int x = tx; x < xmax; x += 8) {
                    __m256 acc = _mm256_setzero_ps();
                    for (int ky = 0; ky < K; ky++) {
                        const float* row = in + (y + ky) * PW + x;
                        for (int kx = 0; kx < K; kx++) {
                            __m256 wv = _mm256_set1_ps(ker[ky * K + kx]);
                            acc = _mm256_fmadd_ps(_mm256_loadu_ps(row + kx), wv, acc);
                        }
                    }
                    _mm256_storeu_ps(out + y * W + x, acc);
                }
            }
        }
    }
}

void conv_optimized(const float* in, float* out, const float* ker, int H, int W, int K) {
    if (K == 3) conv_optimized_k3(in, out, ker, H, W);
    else        conv_optimized_generic(in, out, ker, H, W, K);
}
