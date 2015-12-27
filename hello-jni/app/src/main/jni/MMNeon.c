//
// Created by kunling on 12/25/15.
//

#include <arm_neon.h>
#include "MMUtils.h"
#include "MMNeon.h"


STATIC inline __attribute__((always_inline))  int8_t radd_neon_int8x16(int8x16_t v)  {
    int8_t res=0;
#if defined(__aarch64__)
    // AArch64 new intrinsics
    res = vaddvq_s8(v);
#else
    int64x2_t res_2 = vpaddlq_s32(vpaddlq_s16(vpaddlq_s8(v)));
    res = (int8_t)vadd_s64(vget_high_s64(res_2), vget_low_s64(res_2));
#endif
    return res;
}

// Optimized for matA Matrix, matB vector
STATIC void mmNeonMVector_innerSIMD(int8_t *__restrict__ matA, int8_t *__restrict__ matB,
                                    int8_t *__restrict__ matC, int D_M, int D_N, int D_K) {
    int i,j,k;
    for (i = 0; i < D_M; i++) {
        for (j = 0; j < D_N; j++) {
            int8x16_t v_res_16 = vdupq_n_s8(0);
            for (k=0; k <= D_K-15; k+=16) {
                int8x16_t v_A_16 =  vld1q_s8(&matA[i*D_K + k]);
                int8x16_t v_B_16 =  vld1q_s8(&matB[j*D_N + k]);
                v_res_16 = vmlaq_s8(v_res_16, v_A_16, v_B_16);
            }
            int8_t res = radd_neon_int8x16(v_res_16);
            for(; k < D_K; k++) { //cleanup loop
                res += matA[i*D_K + k] * matB[j*D_N + k];
            }
            matC[i*D_N + j] = res;
        }
    }
}


// TODO: optimize the loop
STATIC void mmNeonMM_interchangeSIMD(int8_t *__restrict__ matA, int8_t *__restrict__ matB,
                                     int8_t *__restrict__ matC, int D_M, int D_N, int D_K) {
    int i,j,k;
    for (i = 0; i < D_M; i++) {
        for (k=0; k < D_K; k++) {
            int8_t temp = matA[i*D_K + k];
            for (j = 0; j < D_N; j++) {
                matC[i*D_N+j] += temp * matB[k*D_N + j];
            }
        }
    }
}

// TODO: add unit test for mmNeon
void mmNeon(int8_t* __restrict__ matA, int8_t* __restrict__ matB, int8_t* __restrict__ matC,
            int D_M, int D_N, int D_K) {

    if (D_N == 1) {
        mmNeonMVector_innerSIMD(matA, matB, matC, D_M, D_N, D_K);
    } else {
        mmNeonMM_interchangeSIMD(matA, matB, matC, D_M, D_N, D_K);
    }

}