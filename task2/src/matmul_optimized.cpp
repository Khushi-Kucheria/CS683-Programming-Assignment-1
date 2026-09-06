#include <immintrin.h>
#include <algorithm>
#include "matmul.h"

static inline float hsum_avx(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 sum128 = _mm_add_ps(lo, hi);
    sum128 = _mm_hadd_ps(sum128, sum128);
    sum128 = _mm_hadd_ps(sum128, sum128);
    return _mm_cvtss_f32(sum128);
}

void matmul_optimized(const float* A, const float* B, float* C,
                      int M, int N, int K, int lda, int ldb, int ldc) {
    constexpr int MC = 64;
    constexpr int NC = 64;
    constexpr int KC = 256;
    constexpr int PREFETCH_DIST = 16;

    for (int i0 = 0; i0 < M; i0 += MC) {
        int i_max = std::min(i0 + MC, M);

        for (int j0 = 0; j0 < N; j0 += NC) {
            int j_max = std::min(j0 + NC, N);

            for (int i = i0; i < i_max; i += 4) {
                int i_end = std::min(i + 4, i_max);

                for (int j = j0; j < j_max; j += 2) {
                    int j_end = std::min(j + 2, j_max);

                    if (i_end - i == 4 && j_end - j == 2) {
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

                        for (int k0 = 0; k0 < K; k0 += KC) {
                            int k_max = std::min(k0 + KC, K);

                            int p = k0;

                            for (; p <= k_max - 8; p += 8) {
                                _mm_prefetch(
                                    reinterpret_cast<const char*>(a0 + p + PREFETCH_DIST),
                                    _MM_HINT_T0);
                                _mm_prefetch(
                                    reinterpret_cast<const char*>(a1 + p + PREFETCH_DIST),
                                    _MM_HINT_T0);
                                _mm_prefetch(
                                    reinterpret_cast<const char*>(a2 + p + PREFETCH_DIST),
                                    _MM_HINT_T0);
                                _mm_prefetch(
                                    reinterpret_cast<const char*>(a3 + p + PREFETCH_DIST),
                                    _MM_HINT_T0);
                                _mm_prefetch(
                                    reinterpret_cast<const char*>(b0 + p + PREFETCH_DIST),
                                    _MM_HINT_T0);
                                _mm_prefetch(
                                    reinterpret_cast<const char*>(b1 + p + PREFETCH_DIST),
                                    _MM_HINT_T0);

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

                            for (; p < k_max; ++p) {
                                float av0 = a0[p];
                                float av1 = a1[p];
                                float av2 = a2[p];
                                float av3 = a3[p];

                                float bv0 = b0[p];
                                float bv1 = b1[p];

                                acc00 = _mm256_add_ps(
                                    acc00,
                                    _mm256_set_ps(
                                        av0 * bv0, 0.0f, 0.0f, 0.0f,
                                        0.0f, 0.0f, 0.0f, 0.0f));

                                acc01 = _mm256_add_ps(
                                    acc01,
                                    _mm256_set_ps(
                                        av0 * bv1, 0.0f, 0.0f, 0.0f,
                                        0.0f, 0.0f, 0.0f, 0.0f));

                                acc10 = _mm256_add_ps(
                                    acc10,
                                    _mm256_set_ps(
                                        av1 * bv0, 0.0f, 0.0f, 0.0f,
                                        0.0f, 0.0f, 0.0f, 0.0f));

                                acc11 = _mm256_add_ps(
                                    acc11,
                                    _mm256_set_ps(
                                        av1 * bv1, 0.0f, 0.0f, 0.0f,
                                        0.0f, 0.0f, 0.0f, 0.0f));

                                acc20 = _mm256_add_ps(
                                    acc20,
                                    _mm256_set_ps(
                                        av2 * bv0, 0.0f, 0.0f, 0.0f,
                                        0.0f, 0.0f, 0.0f, 0.0f));

                                acc21 = _mm256_add_ps(
                                    acc21,
                                    _mm256_set_ps(
                                        av2 * bv1, 0.0f, 0.0f, 0.0f,
                                        0.0f, 0.0f, 0.0f, 0.0f));

                                acc30 = _mm256_add_ps(
                                    acc30,
                                    _mm256_set_ps(
                                        av3 * bv0, 0.0f, 0.0f, 0.0f,
                                        0.0f, 0.0f, 0.0f, 0.0f));

                                acc31 = _mm256_add_ps(
                                    acc31,
                                    _mm256_set_ps(
                                        av3 * bv1, 0.0f, 0.0f, 0.0f,
                                        0.0f, 0.0f, 0.0f, 0.0f));
                            }
                        }

                        C[static_cast<long>(i + 0) * ldc + j + 0] = hsum_avx(acc00);
                        C[static_cast<long>(i + 0) * ldc + j + 1] = hsum_avx(acc01);
                        C[static_cast<long>(i + 1) * ldc + j + 0] = hsum_avx(acc10);
                        C[static_cast<long>(i + 1) * ldc + j + 1] = hsum_avx(acc11);
                        C[static_cast<long>(i + 2) * ldc + j + 0] = hsum_avx(acc20);
                        C[static_cast<long>(i + 2) * ldc + j + 1] = hsum_avx(acc21);
                        C[static_cast<long>(i + 3) * ldc + j + 0] = hsum_avx(acc30);
                        C[static_cast<long>(i + 3) * ldc + j + 1] = hsum_avx(acc31);
                    } else {
                        for (int ii = i; ii < i_end; ++ii) {
                            for (int jj = j; jj < j_end; ++jj) {
                                const float* a = A + static_cast<long>(ii) * lda;
                                const float* b = B + static_cast<long>(jj) * ldb;

                                float acc = 0.0f;

                                for (int k0 = 0; k0 < K; k0 += KC) {
                                    int k_max = std::min(k0 + KC, K);

                                    for (int p = k0; p < k_max; ++p) {
                                        if (p + PREFETCH_DIST < k_max) {
                                            _mm_prefetch(
                                                reinterpret_cast<const char*>(
                                                    a + p + PREFETCH_DIST),
                                                _MM_HINT_T0);
                                            _mm_prefetch(
                                                reinterpret_cast<const char*>(
                                                    b + p + PREFETCH_DIST),
                                                _MM_HINT_T0);
                                        }

                                        acc += a[p] * b[p];
                                    }
                                }

                                C[static_cast<long>(ii) * ldc + jj] = acc;
                            }
                        }
                    }
                }
            }
        }
    }
}
