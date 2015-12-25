//
// Created by kunling on 12/25/15.
//

#include "MMScalar.h"



void mmScalar(int8_t* __restrict__ A, int8_t* __restrict__ B, int8_t* __restrict__ C,
              int D_M, int D_N, int D_K) {
    int i,j,k;
    for (i = 0; i < D_M; i++) {
        for (j = 0; j < D_N; j++) {
            int8_t res = 0;
            for (k=0; k < D_K; k++) {
                res += A[i*D_K + k] * B[k*D_N + j];
            }
            C[i*D_N + j] = res;
        }
    }
}