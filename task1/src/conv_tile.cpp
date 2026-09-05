// conv_tile.cpp  STAGE 3: CACHE TILING

#include "convolution.h"
#include <algorithm>   

#ifndef TILE_H
#define TILE_H 32
#endif
#ifndef TILE_W
#define TILE_W 32
#endif
void conv_tile(const float* in, float* out, const float* ker,
               int H, int W, int K) {
    const int p = K / 2;
    const int in_stride = W + 2 * p;
      for (int ty = 0; ty < H; ty += TILE_H) { //H and W are divided into TILE X TILE blocks hence both ty and tx jump by tile size.
       
            int oy_end = std::min(ty + TILE_H, H); //tile size might not divide H and W evenly, so in such a case, take the min of ty/tx + tile and the row/col number.
         for (int tx = 0; tx < W; tx += TILE_W) {
            int ox_end = std::min(tx + TILE_W, W);
        
      
              for (int oy = ty; oy < oy_end; ++oy) { //entire block placed in the cache, hence when the kernel matrix traverses, all the block data would not be kicked out and by the time the kernel accesses say the next row, it gets a hit.
                 for (int ox = tx; ox < ox_end; ++ox) {
                          float acc = 0.0f; //remaining logic is the same as that of conv_naive.cpp
                          for (int ky = 0; ky < K; ++ky) {
                              for (int kx = 0; kx < K; ++kx) {
                                  acc += in[(oy + ky) * in_stride + (ox + kx)]
                                       * ker[ky * K + kx];
                              }
                          }
                          out[oy * W + ox] = acc;
                 }
              }
        }
      }
}
