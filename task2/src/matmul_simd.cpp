#include <immintrin.h>

#include "matmul.h"

static inline float hsum_avx(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 sum128 = _mm_add_ps(lo, hi);
    sum128 = _mm_hadd_ps(sum128, sum128);
    sum128 = _mm_hadd_ps(sum128, sum128);
    return _mm_cvtss_f32(sum128);
}

void matmul_simd(const float* A, const float* B, float* C,
                 int M, int N, int K, int lda, int ldb, int ldc) {
    int i = 0;
    for (; i <= M - 4; i += 4) {
        int j = 0;
        for (; j <= N - 2; j += 2) {
            __m256 acc00 = _mm256_setzero_ps();
            __m256 acc01 = _mm256_setzero_ps();
            __m256 acc10 = _mm256_setzero_ps();
            __m256 acc11 = _mm256_setzero_ps();
            __m256 acc20 = _mm256_setzero_ps();
            __m256 acc21 = _mm256_setzero_ps();
            __m256 acc30 = _mm256_setzero_ps();
            __m256 acc31 = _mm256_setzero_ps();

            const float* a0 = A + static_cast<long>(i + 0) * lda;
            const float* a1 = A + static_cast<long>(i + 1) * lda;
            const float* a2 = A + static_cast<long>(i + 2) * lda;
            const float* a3 = A + static_cast<long>(i + 3) * lda;

            const float* b0 = B + static_cast<long>(j + 0) * ldb;
            const float* b1 = B + static_cast<long>(j + 1) * ldb;

            int p = 0;
            for (; p <= K - 8; p += 8) {
                __m256 vb0 = _mm256_loadu_ps(b0 + p);
                __m256 vb1 = _mm256_loadu_ps(b1 + p);

                __m256 va0 = _mm256_loadu_ps(a0 + p);
                acc00 = _mm256_fmadd_ps(va0, vb0, acc00);
                acc01 = _mm256_fmadd_ps(va0, vb1, acc01);

                __m256 va1 = _mm256_loadu_ps(a1 + p);
                acc10 = _mm256_fmadd_ps(va1, vb0, acc10);
                acc11 = _mm256_fmadd_ps(va1, vb1, acc11);

                __m256 va2 = _mm256_loadu_ps(a2 + p);
                acc20 = _mm256_fmadd_ps(va2, vb0, acc20);
                acc21 = _mm256_fmadd_ps(va2, vb1, acc21);

                __m256 va3 = _mm256_loadu_ps(a3 + p);
                acc30 = _mm256_fmadd_ps(va3, vb0, acc30);
                acc31 = _mm256_fmadd_ps(va3, vb1, acc31);
            }

            float sum00 = hsum_avx(acc00);
            float sum01 = hsum_avx(acc01);
            float sum10 = hsum_avx(acc10);
            float sum11 = hsum_avx(acc11);
            float sum20 = hsum_avx(acc20);
            float sum21 = hsum_avx(acc21);
            float sum30 = hsum_avx(acc30);
            float sum31 = hsum_avx(acc31);

            for (; p < K; ++p) {
                sum00 += a0[p] * b0[p];
                sum01 += a0[p] * b1[p];
                sum10 += a1[p] * b0[p];
                sum11 += a1[p] * b1[p];
                sum20 += a2[p] * b0[p];
                sum21 += a2[p] * b1[p];
                sum30 += a3[p] * b0[p];
                sum31 += a3[p] * b1[p];
            }

            C[static_cast<long>(i + 0) * ldc + (j + 0)] = sum00;
            C[static_cast<long>(i + 0) * ldc + (j + 1)] = sum01;
            C[static_cast<long>(i + 1) * ldc + (j + 0)] = sum10;
            C[static_cast<long>(i + 1) * ldc + (j + 1)] = sum11;
            C[static_cast<long>(i + 2) * ldc + (j + 0)] = sum20;
            C[static_cast<long>(i + 2) * ldc + (j + 1)] = sum21;
            C[static_cast<long>(i + 3) * ldc + (j + 0)] = sum30;
            C[static_cast<long>(i + 3) * ldc + (j + 1)] = sum31;
        }

        for (; j < N; ++j) {
            for (int sub_i = i; sub_i < i + 4; ++sub_i) {
                float acc = 0.0f;
                const float* a = A + static_cast<long>(sub_i) * lda;
                const float* b = B + static_cast<long>(j) * ldb;
                for (int p = 0; p < K; ++p) {
                    acc += a[p] * b[p];
                }
                C[static_cast<long>(sub_i) * ldc + j] = acc;
            }
        }
    }

    for (; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float acc = 0.0f;
            const float* a = A + static_cast<long>(i) * lda;
            const float* b = B + static_cast<long>(j) * ldb;
            for (int p = 0; p < K; ++p) {
                acc += a[p] * b[p];
            }
            C[static_cast<long>(i) * ldc + j] = acc;
        }
    }
}
