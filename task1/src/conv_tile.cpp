void conv_tile(const float* in, float* out, const float* ker,
               int H, int W, int K) {
    const int p = K / 2;
    const int in_stride = W + 2 * p;

    for (int ty = 0; ty < H; ty += TILE) {
        int oy_end = std::min(ty + TILE, H);
        for (int tx = 0; tx < W; tx += TILE) {
            int ox_end = std::min(tx + TILE, W);

            for (int oy = ty; oy < oy_end; ++oy) {
                for (int ox = tx; ox < ox_end; ++ox) {
                    float acc = 0.0f;
                    for (int ky = 0; ky < K; ++ky) {
                        const float* in_row = in + (oy + ky) * in_stride + ox;  // computed ONCE per ky
                        const float* ker_row = ker + ky * K;
                        for (int kx = 0; kx < K; ++kx) {
                            acc += in_row[kx] * ker_row[kx];   // pure pointer offset, no multiply
                        }
                    }
                    out[oy * W + ox] = acc;
                }
            }
        }
    }
}
