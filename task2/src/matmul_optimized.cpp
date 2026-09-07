#include <immintrin.h>
#include <algorithm>
#include "matmul.h"

static inline float hsum_avx(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 s = _mm_add_ps(lo, hi);
    s = _mm_hadd_ps(s, s);
    s = _mm_hadd_ps(s, s);
    return _mm_cvtss_f32(s);
}

void matmul_optimized(const float* A, const float* B, float* C,
                      int M, int N, int K,
                      int lda, int ldb, int ldc) {

    constexpr int MC = 64;
    constexpr int NC = 64;
    constexpr int KC = 256;
    constexpr int PREFETCH_DIST = 32;

    for (int i0 = 0; i0 < M; i0 += MC) {
        int imax = std::min(i0 + MC, M);

        for (int j0 = 0; j0 < N; j0 += NC) {
            int jmax = std::min(j0 + NC, N);

            for (int i = i0; i < imax; i += 4) {
                int im = std::min(i + 4, imax);

                for (int j = j0; j < jmax; j += 2) {
                    int jm = std::min(j + 2, jmax);

                    if (im - i < 4 || jm - j < 2) {
                        for (int ii = i; ii < im; ++ii) {
                            for (int jj = j; jj < jm; ++jj) {
                                float acc = 0.0f;

                                const float* a =
                                    A + static_cast<long>(ii) * lda;

                                const float* b =
                                    B + static_cast<long>(jj) * ldb;

                                for (int k = 0; k < K; ++k)
                                    acc += a[k] * b[k];

                                C[static_cast<long>(ii) * ldc + jj] = acc;
                            }
                        }
                        continue;
                    }

                    __m256 acc00 = _mm256_setzero_ps();
                    __m256 acc01 = _mm256_setzero_ps();
                    __m256 acc10 = _mm256_setzero_ps();
                    __m256 acc11 = _mm256_setzero_ps();
                    __m256 acc20 = _mm256_setzero_ps();
                    __m256 acc21 = _mm256_setzero_ps();
                    __m256 acc30 = _mm256_setzero_ps();
                    __m256 acc31 = _mm256_setzero_ps();

                    const float* a0 =
                        A + static_cast<long>(i + 0) * lda;
                    const float* a1 =
                        A + static_cast<long>(i + 1) * lda;
                    const float* a2 =
                        A + static_cast<long>(i + 2) * lda;
                    const float* a3 =
                        A + static_cast<long>(i + 3) * lda;

                    const float* b0 =
                        B + static_cast<long>(j + 0) * ldb;
                    const float* b1 =
                        B + static_cast<long>(j + 1) * ldb;

                    for (int k0 = 0; k0 < K; k0 += KC) {
                        int kend = std::min(k0 + KC, K);

                        int k = k0;

                        for (; k <= kend - 8; k += 8) {

                            if (k + PREFETCH_DIST < kend) {
                                _mm_prefetch(
                                    reinterpret_cast<const char*>(
                                        a0 + k + PREFETCH_DIST),
                                    _MM_HINT_T0);

                                _mm_prefetch(
                                    reinterpret_cast<const char*>(
                                        a1 + k + PREFETCH_DIST),
                                    _MM_HINT_T0);

                                _mm_prefetch(
                                    reinterpret_cast<const char*>(
                                        a2 + k + PREFETCH_DIST),
                                    _MM_HINT_T0);

                                _mm_prefetch(
                                    reinterpret_cast<const char*>(
                                        a3 + k + PREFETCH_DIST),
                                    _MM_HINT_T0);

                                _mm_prefetch(
                                    reinterpret_cast<const char*>(
                                        b0 + k + PREFETCH_DIST),
                                    _MM_HINT_T0);

                                _mm_prefetch(
                                    reinterpret_cast<const char*>(
                                        b1 + k + PREFETCH_DIST),
                                    _MM_HINT_T0);
                            }

                            __m256 vb0 =
                                _mm256_loadu_ps(b0 + k);
                            __m256 vb1 =
                                _mm256_loadu_ps(b1 + k);

                            __m256 va0 =
                                _mm256_loadu_ps(a0 + k);
                            __m256 va1 =
                                _mm256_loadu_ps(a1 + k);
                            __m256 va2 =
                                _mm256_loadu_ps(a2 + k);
                            __m256 va3 =
                                _mm256_loadu_ps(a3 + k);

                            acc00 = _mm256_fmadd_ps(va0, vb0, acc00);
                            acc01 = _mm256_fmadd_ps(va0, vb1, acc01);

                            acc10 = _mm256_fmadd_ps(va1, vb0, acc10);
                            acc11 = _mm256_fmadd_ps(va1, vb1, acc11);

                            acc20 = _mm256_fmadd_ps(va2, vb0, acc20);
                            acc21 = _mm256_fmadd_ps(va2, vb1, acc21);

                            acc30 = _mm256_fmadd_ps(va3, vb0, acc30);
                            acc31 = _mm256_fmadd_ps(va3, vb1, acc31);
                        }

                        for (; k < kend; ++k) {
                            float bv0 = b0[k];
                            float bv1 = b1[k];

                            acc00 = _mm256_add_ps(
                                acc00,
                                _mm256_set1_ps(a0[k] * bv0));

                            acc01 = _mm256_add_ps(
                                acc01,
                                _mm256_set1_ps(a0[k] * bv1));

                            acc10 = _mm256_add_ps(
                                acc10,
                                _mm256_set1_ps(a1[k] * bv0));

                            acc11 = _mm256_add_ps(
                                acc11,
                                _mm256_set1_ps(a1[k] * bv1));

                            acc20 = _mm256_add_ps(
                                acc20,
                                _mm256_set1_ps(a2[k] * bv0));

                            acc21 = _mm256_add_ps(
                                acc21,
                                _mm256_set1_ps(a2[k] * bv1));

                            acc30 = _mm256_add_ps(
                                acc30,
                                _mm256_set1_ps(a3[k] * bv0));

                            acc31 = _mm256_add_ps(
                                acc31,
                                _mm256_set1_ps(a3[k] * bv1));
                        }
                    }

                    C[static_cast<long>(i + 0) * ldc + j + 0] =
                        hsum_avx(acc00);
                    C[static_cast<long>(i + 0) * ldc + j + 1] =
                        hsum_avx(acc01);

                    C[static_cast<long>(i + 1) * ldc + j + 0] =
                        hsum_avx(acc10);
                    C[static_cast<long>(i + 1) * ldc + j + 1] =
                        hsum_avx(acc11);

                    C[static_cast<long>(i + 2) * ldc + j + 0] =
                        hsum_avx(acc20);
                    C[static_cast<long>(i + 2) * ldc + j + 1] =
                        hsum_avx(acc21);

                    C[static_cast<long>(i + 3) * ldc + j + 0] =
                        hsum_avx(acc30);
                    C[static_cast<long>(i + 3) * ldc + j + 1] =
                        hsum_avx(acc31);
                }
            }
        }
    }
}
