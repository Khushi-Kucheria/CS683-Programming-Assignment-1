void conv_simd(const float* in, float* out, const float* ker,
               int H, int W, int K) {
    const int p = K / 2;
    const int in_stride = W + 2 * p;

    // Precompute broadcasts once (kernel doesn't change across the whole image)
    static thread_local std::vector<__m256> ker_bcast;
    ker_bcast.resize(K * K);
    for (int i = 0; i < K * K; ++i) ker_bcast[i] = _mm256_set1_ps(ker[i]);

    for (int oy = 0; oy < H; ++oy) {
        int ox = 0;
        for (; ox + 8 <= W; ox += 8) {
            __m256 acc = _mm256_setzero_ps();
            for (int ky = 0; ky < K; ++ky) {
                for (int kx = 0; kx < K; ++kx) {
                    const float* row_ptr = &in[(oy + ky) * in_stride + (ox + kx)];
                    __m256 in_vec = _mm256_loadu_ps(row_ptr);
                    acc = _mm256_fmadd_ps(in_vec, ker_bcast[ky * K + kx], acc);
                }
            }
            _mm256_storeu_ps(&out[oy * W + ox], acc);
        }
        for (; ox < W; ++ox) {
            float acc = 0.0f;
            for (int ky = 0; ky < K; ++ky)
                for (int kx = 0; kx < K; ++kx)
                    acc += in[(oy + ky) * in_stride + (ox + kx)] * ker[ky * K + kx];
            out[oy * W + ox] = acc;
        }
    }
}
