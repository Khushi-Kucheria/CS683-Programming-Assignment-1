// conv_unroll.cpp  STAGE 2: LOOP UNROLLING
#include "convolution.h"

void conv_unroll(const float* in, float* out, const float* ker,
                 int H, int W, int K) {
    const int p = K / 2;
    const int in_stride = W + 2 * p;  // padded row stride

    // so in this, we are calculating and updating 4 values in the same loop
    // why this would help is - if its a single update then we have to update the loop counter and check for index bounds everytime
    // now all that will reduce by 4 times
    // like first the ox loop was updating W times (loop counter and stuff) and now its updating W/4 times so that time is saved
  
    for (int oy = 0; oy < H; ++oy) {
        for (int ox = 0; ox < W; ox+=4) {
            float acc0 = 0.0f;
            float acc1 = 0.0f;
            float acc2 = 0.0f;
            float acc3 = 0.0f;
            for (int ky = 0; ky < K; ++ky) {
                for (int kx = 0; kx < K; ++kx) {
                    float w = ker[ky * K + kx];
                    acc0 += w * in[(oy + ky) * in_stride + ox + kx + 0];
                    acc1 += w * in[(oy + ky) * in_stride + ox + kx + 1];
                    acc2 += w * in[(oy + ky) * in_stride + ox + kx + 2];
                    acc3 += w * in[(oy + ky) * in_stride + ox + kx + 3];
                }
            }
            out[oy * W + ox + 0] = acc0;
            out[oy * W + ox + 1] = acc1;
            out[oy * W + ox + 2] = acc2;
            out[oy * W + ox + 3] = acc3;
        }
    }
    // conv_naive(in, out, ker, H, W, K);
}
