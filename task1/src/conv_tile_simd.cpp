// conv_tiled_simd.c
// 2D convolution (3x3 kernel, "valid" mode) combining TILING (cache blocking)
// with SIMD (AVX2 + FMA) vectorization.
//
// Build:
//   gcc -O2 -mavx2 -mfma conv_tiled_simd.c -o conv_tiled_simd -lm
// Run:
//   ./conv_tiled_simd <N>          e.g. ./conv_tiled_simd 2048
//
// Prints the best-of-REPS runtime in seconds. Compare this against your
// existing "no optimization" naive runtime for the same N to get the
// normalized speedup for the "Tiling + SIMD" bar.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <immintrin.h>

#define KSIZE 3
#define TILE 64      // tune this: try 32/64/128 and keep whichever is fastest
#define REPS 7

static float kernel3x3[KSIZE][KSIZE] = {
    {0.05f, 0.10f, 0.05f},
    {0.10f, 0.40f, 0.10f},
    {0.05f, 0.10f, 0.05f}
};

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

// ---- naive (baseline, for correctness check only) ----
static void conv_naive(const float* restrict in, float* restrict out, int N) {
    int OW = N - KSIZE + 1;
    for (int i = 0; i < OW; i++)
        for (int j = 0; j < OW; j++) {
            float acc = 0.0f;
            for (int ki = 0; ki < KSIZE; ki++)
                for (int kj = 0; kj < KSIZE; kj++)
                    acc += in[(i + ki) * N + (j + kj)] * kernel3x3[ki][kj];
            out[i * OW + j] = acc;
        }
}

// ---- Tiling + SIMD combined ----
static void conv_tiled_simd(const float* restrict in, float* restrict out, int N) {
    int OW = N - KSIZE + 1;
    const float k00=kernel3x3[0][0], k01=kernel3x3[0][1], k02=kernel3x3[0][2];
    const float k10=kernel3x3[1][0], k11=kernel3x3[1][1], k12=kernel3x3[1][2];
    const float k20=kernel3x3[2][0], k21=kernel3x3[2][1], k22=kernel3x3[2][2];
    __m256 v00=_mm256_set1_ps(k00), v01=_mm256_set1_ps(k01), v02=_mm256_set1_ps(k02);
    __m256 v10=_mm256_set1_ps(k10), v11=_mm256_set1_ps(k11), v12=_mm256_set1_ps(k12);
    __m256 v20=_mm256_set1_ps(k20), v21=_mm256_set1_ps(k21), v22=_mm256_set1_ps(k22);

    // Outer loops walk TILE x TILE blocks of the OUTPUT -> cache blocking.
    for (int ii = 0; ii < OW; ii += TILE) {
        int imax = ii + TILE < OW ? ii + TILE : OW;
        for (int jj = 0; jj < OW; jj += TILE) {
            int jmax = jj + TILE < OW ? jj + TILE : OW;
            // Inner loops: vectorized 8-wide FMA sweep within the tile -> SIMD.
            for (int i = ii; i < imax; i++) {
                const float* r0 = in + (i + 0) * N;
                const float* r1 = in + (i + 1) * N;
                const float* r2 = in + (i + 2) * N;
                int j = jj;
                for (; j <= jmax - 8; j += 8) {
                    __m256 acc = _mm256_mul_ps(_mm256_loadu_ps(r0 + j), v00);
                    acc = _mm256_fmadd_ps(_mm256_loadu_ps(r0 + j + 1), v01, acc);
                    acc = _mm256_fmadd_ps(_mm256_loadu_ps(r0 + j + 2), v02, acc);
                    acc = _mm256_fmadd_ps(_mm256_loadu_ps(r1 + j), v10, acc);
                    acc = _mm256_fmadd_ps(_mm256_loadu_ps(r1 + j + 1), v11, acc);
                    acc = _mm256_fmadd_ps(_mm256_loadu_ps(r1 + j + 2), v12, acc);
                    acc = _mm256_fmadd_ps(_mm256_loadu_ps(r2 + j), v20, acc);
                    acc = _mm256_fmadd_ps(_mm256_loadu_ps(r2 + j + 1), v21, acc);
                    acc = _mm256_fmadd_ps(_mm256_loadu_ps(r2 + j + 2), v22, acc);
                    _mm256_storeu_ps(out + i * OW + j, acc);
                }
                for (; j < jmax; j++) {   // remainder columns in this tile
                    out[i * OW + j] =
                        r0[j]*k00 + r0[j+1]*k01 + r0[j+2]*k02 +
                        r1[j]*k10 + r1[j+1]*k11 + r1[j+2]*k12 +
                        r2[j]*k20 + r2[j+1]*k21 + r2[j+2]*k22;
                }
            }
        }
    }
}

int main(int argc, char** argv) {
    int N = argc > 1 ? atoi(argv[1]) : 2048;
    int OW = N - KSIZE + 1;
    size_t in_elems = (size_t)N * N, out_elems = (size_t)OW * OW;

    float* in = malloc(sizeof(float) * in_elems);
    float* out_ref = malloc(sizeof(float) * out_elems);
    float* out = malloc(sizeof(float) * out_elems);

    srand(42);
    for (size_t i = 0; i < in_elems; i++) in[i] = (float)(rand() % 1000) / 1000.0f;

    conv_naive(in, out_ref, N);   // correctness reference

    double best = 1e18;
    for (int r = 0; r < REPS; r++) {
        double t0 = now_sec();
        conv_tiled_simd(in, out, N);
        double t1 = now_sec();
        if (t1 - t0 < best) best = t1 - t0;
    }

    double maxdiff = 0.0;
    for (size_t i = 0; i < out_elems; i++) {
        double d = fabs((double)out[i] - (double)out_ref[i]);
        if (d > maxdiff) maxdiff = d;
    }

    printf("N=%d  tiled_simd_time_s=%.6f  correctness_maxdiff=%g\n", N, best, maxdiff);
    if (maxdiff > 1e-3) fprintf(stderr, "WARNING: output mismatch vs naive!\n");

    free(in); free(out_ref); free(out);
    return 0;
}
