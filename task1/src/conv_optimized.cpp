#include <immintrin.h>
#include <algorithm>
#include "convolution.h"

#ifndef TILE
#define TILE 32
#endif
static_assert(TILE % 8 == 0, "TILE must be a multiple of 8 to keep every tile 8-aligned");

void conv_optimized(const float* in, float* out, const float* ker, int H, int W, int K) {
    const int p = K / 2;
    const int PW = W + 2 * p;

    for (int ty = 0; ty < H; ty += TILE) { //tiling is done here
        int ymax = std::min(ty + TILE, H);
        for (int tx = 0; tx < W; tx += TILE) {
            int xmax = std::min(tx + TILE, W);
            for (int y = ty; y < ymax; y++) {
                for (int x = tx; x < xmax; x += 8) { //inside the tiling, vectorize 8 pixels at once.
                    __m256 acc = _mm256_setzero_ps();
                    for (int ky = 0; ky < K; ky++) {
                        const float* row = in + (y + ky) * PW + x;
                        for (int kx = 0; kx < K; kx++) {
                            __m256 wv = _mm256_set1_ps(ker[ky * K + kx]); //store the same kernel weight copy across all 8 lanes so as to avoid repeated loading
                            acc = _mm256_fmadd_ps(_mm256_loadu_ps(row + kx), wv, acc); //fuse multiply and add across the 8 lanes parallely so that its faster.
                        }
                    }
                    _mm256_storeu_ps(out + y * W + x, acc); //finally write the 8 finished output pixls back.
                }
            }
        }
    }
}
