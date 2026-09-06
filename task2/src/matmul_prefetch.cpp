#include <immintrin.h>
#include <algorithm>
#include "matmul.h"

void matmul_prefetch(const float* A, const float* B, float* C,
                     int M, int N, int K, int lda, int ldb, int ldc) {
    constexpr int MC = 64;
    constexpr int NC = 64;
    constexpr int KC = 256;
    
    constexpr int PREFETCH_DIST = 16; 

    for (int i0 = 0; i0 < M; i0 += MC) {
        int i_max = std::min(i0 + MC, M);
        for (int j0 = 0; j0 < N; j0 += NC) {
            int j_max = std::min(j0 + NC, N);
            for (int k0 = 0; k0 < K; k0 += KC) {
                int k_max = std::min(k0 + KC, K);

                for (int i = i0; i < i_max; ++i) {
                    const float* a_row = A + i * static_cast<long>(lda);
                    for (int j = j0; j < j_max; ++j) {
                        const float* b_row = B + j * static_cast<long>(ldb);
                        
                        float acc = (k0 == 0) ? 0.0f : C[i * static_cast<long>(ldc) + j];

                        int p = k0;
                        for (; p <= k_max - 16; p += 16) {
                            _mm_prefetch(reinterpret_cast<const char*>(a_row + p + PREFETCH_DIST), _MM_HINT_T0);
                            _mm_prefetch(reinterpret_cast<const char*>(b_row + p + PREFETCH_DIST), _MM_HINT_T0);

                            for (int u = 0; u < 16; ++u) {
                                acc += a_row[p + u] * b_row[p + u];
                            }
                        }

                        for (; p < k_max; ++p) {
                            acc += a_row[p] * b_row[p];
                        }

                        C[i * static_cast<long>(ldc) + j] = acc;
                    }
                }
            }
        }
    }
}
