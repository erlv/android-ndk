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
            int8x16_t v_res_16 = vdupq_n_s8(0);
            for (k=0; k <= D_K-15; k+=16) {
                int8x16_t v_A_16 =  vld1q_s8(&matA[i*D_K + k]);
                int8x16_t v_B_16 =  vld1q_s8(&matB[k]);
                v_res_16 = vmlaq_s8(v_res_16, v_A_16, v_B_16);
            }
            int8_t res = radd_neon_int8x16(v_res_16);
            for(; k < D_K; k++) { //cleanup loop
                res += matA[i*D_K + k] * matB[k];
            }
            matC[i*D_N] = res;
    }
}

// v0: interchange the loop of j and k
STATIC void mmNeonMM_interchangeSIMD_v_m0(int8_t *__restrict__ matA, int8_t *__restrict__ matB,
                                        int8_t *__restrict__ matC, int D_M, int D_N, int D_K) {
    int i,j,k;
    for (i = 0; i < D_M; i++) {
        for (j = 0; j < D_N; j++) {
            int8_t res = 0;
            for (k=0; k < D_K; k++) {
                res += matA[i*D_K + k] * matB[k*D_N + j];
            }
            matC[i*D_N + j] = res;
        }
    }
}

// v0: interchange the loop of j and k
STATIC void mmNeonMM_interchangeSIMD_v0(int8_t *__restrict__ matA, int8_t *__restrict__ matB,
                                        int8_t *__restrict__ matC, int D_M, int D_N, int D_K) {
    int i, j, k;
    for (i = 0; i < D_M; i++) {
        for (k = 0; k <= D_K; k++) {
            int8_t temp = matA[i * D_K + k];
            for (j = 0; j < D_N; j++) {
                matC[i * D_N + j] += temp * matB[k * D_N + j];
            }
        }
    }
}

// v1: vectorize the inner j loop
STATIC void mmNeonMM_interchangeSIMD_v1(int8_t *__restrict__ matA, int8_t *__restrict__ matB,
                                        int8_t *__restrict__ matC, int D_M, int D_N, int D_K) {
    int i, j, k;
    for (i = 0; i < D_M; i++) {
        for (k = 0; k < D_K; k ++) {
            int8_t temp = matA[i * D_K + k];
            int8x16_t v_temp = vdupq_n_s8(temp);
            for (j = 0; j <= D_N - 15; j += 16) {
                int8x16_t v_C_16 = vld1q_s8(&matC[i * D_N + j]);
                int8x16_t v_B_16 = vld1q_s8(&matB[k * D_N + j]);
                v_C_16 = vmlaq_s8(v_C_16, v_B_16, v_temp);
                vst1q_s8(&matC[i * D_N + j], v_C_16);
            }
            for (; j < D_N; j++) {
                int8_t matC_temp = matC[i * D_N + j];
                matC_temp += temp * matB[k * D_N + j];
                matC[i * D_N + j] = matC_temp;
            }
        }
    }
}


// v2: unroll k loop
STATIC void mmNeonMM_interchangeSIMD_v2(int8_t *__restrict__ matA, int8_t *__restrict__ matB,
                                        int8_t *__restrict__ matC, int D_M, int D_N, int D_K) {
    int i, j, k;
    for (i = 0; i < D_M; i++) {
        for (k = 0; k <= D_K - 3; k += 4) {
            int8_t temp = matA[i * D_K + k];
            int8_t temp_1 = matA[i*D_K + k+1];
            int8_t temp_2 = matA[i*D_K + k+2];
            int8_t temp_3 = matA[i*D_K + k+3];
            for (j = 0; j < D_N; j++) {
                int8_t matC_temp = matC[i*D_N + j];
                matC_temp += temp * matB[k*D_N + j];
                matC_temp += temp_1 * matB[(k+1)*D_N + j];
                matC_temp += temp_2 * matB[(k+2)*D_N + j];
                matC_temp += temp_3 * matB[(k+3)*D_N + j];
                matC[i*D_N + j] = matC_temp;
            }
        }
        for (; k < D_K; k++) {
            int8_t temp = matA[i * D_K + k];
            for (; j < D_N; j++) {
                int8_t matC_temp = matC[i * D_N + j];
                matC_temp += temp * matB[k * D_N + j];
                matC[i * D_N + j] = matC_temp;
            }
        }
    }
}

// v3: vectorize the inner j loop + unroll the outer k loop
STATIC void mmNeonMM_interchangeSIMD_v3(int8_t *__restrict__ matA, int8_t *__restrict__ matB,
                                     int8_t *__restrict__ matC, int D_M, int D_N, int D_K) {
    int i,j,k;
    for (i = 0; i < D_M; i++) {
        for (k=0; k <= D_K-3; k+= 4) {
            int8_t temp = matA[i*D_K + k];
            int8_t temp_1 = matA[i*D_K + k+1];
            int8_t temp_2 = matA[i*D_K + k+2];
            int8_t temp_3 = matA[i*D_K + k+3];
            int8x16_t v_temp = vdupq_n_s8(temp);
            int8x16_t v_temp_1 = vdupq_n_s8(temp_1);
            int8x16_t v_temp_2 = vdupq_n_s8(temp_2);
            int8x16_t v_temp_3 = vdupq_n_s8(temp_3);
            for (j = 0; j <= D_N-15; j+=16) {
                int8x16_t v_C_16 = vld1q_s8(&matC[i * D_N + j]);
                int8x16_t v_B_16 = vld1q_s8(&matB[k * D_N + j]);
                int8x16_t v_B_16_1 = vld1q_s8(&matB[(k+1) * D_N + j]);
                int8x16_t v_B_16_2 = vld1q_s8(&matB[(k+2) * D_N + j]);
                int8x16_t v_B_16_3 = vld1q_s8(&matB[(k+3) * D_N + j]);
                v_C_16 = vmlaq_s8(v_C_16, v_B_16, v_temp);
                v_C_16 = vmlaq_s8(v_C_16, v_B_16_1, v_temp_1);
                v_C_16 = vmlaq_s8(v_C_16, v_B_16_2, v_temp_2);
                v_C_16 = vmlaq_s8(v_C_16, v_B_16_3, v_temp_3);
                vst1q_s8(&matC[i * D_N + j], v_C_16);
            }
            for(; j < D_N; j++) {
                int8_t matC_temp = matC[i*D_N + j];
                matC_temp += temp * matB[k*D_N + j];
                matC_temp += temp_1 * matB[(k+1)*D_N + j];
                matC_temp += temp_2 * matB[(k+2)*D_N + j];
                matC_temp += temp_3 * matB[(k+3)*D_N + j];
                matC[i*D_N + j] = matC_temp;
            }
        }
        for (; k < D_K; k++) {
            int8_t temp = matA[i * D_K + k];
            int8x16_t v_temp = vdupq_n_s8(temp);
            for (j = 0; j <= D_N - 15; j += 16) {
                int8x16_t v_C_16 = vld1q_s8(&matC[i * D_N + j]);
                int8x16_t v_B_16 = vld1q_s8(&matB[k * D_N + j]);
                v_C_16 = vmlaq_s8(v_C_16, v_B_16, v_temp);
                vst1q_s8(&matC[i * D_N + j], v_C_16);
            }
            for (; j < D_N; j++) {
                int8_t matC_temp = matC[i * D_N + j];
                matC_temp += temp * matB[k * D_N + j];
                matC[i * D_N + j] = matC_temp;
            }
        }
    }
}

void mmNeonMM_blocking_kernel_16x16_v1(int8_t* __restrict__ matA, int8_t* __restrict__ matB,
                                    int8_t* __restrict__ matC) {
    int i, j, k;
    for (i = 0; i < 16; i++) {
        for (k = 0; k < 16; k += 4) {
            int8_t temp = matA[i * 16 + k];
            int8_t temp_1 = matA[i * 16 + k + 1];
            int8_t temp_2 = matA[i * 16 + k + 2];
            int8_t temp_3 = matA[i * 16 + k + 3];
            int8x16_t v_temp = vdupq_n_s8(temp);
            int8x16_t v_temp_1 = vdupq_n_s8(temp_1);
            int8x16_t v_temp_2 = vdupq_n_s8(temp_2);
            int8x16_t v_temp_3 = vdupq_n_s8(temp_3);
            for (j = 0; j < 16; j += 16) {
                int8x16_t v_C_16 = vld1q_s8(&matC[i * 16 + j]);
                int8x16_t v_B_16 = vld1q_s8(&matB[k * 16 + j]);
                int8x16_t v_B_16_1 = vld1q_s8(&matB[(k + 1) * 16 + j]);
                int8x16_t v_B_16_2 = vld1q_s8(&matB[(k + 2) * 16 + j]);
                int8x16_t v_B_16_3 = vld1q_s8(&matB[(k + 3) * 16 + j]);
                v_C_16 = vmlaq_s8(v_C_16, v_B_16, v_temp);
                v_C_16 = vmlaq_s8(v_C_16, v_B_16_1, v_temp_1);
                v_C_16 = vmlaq_s8(v_C_16, v_B_16_2, v_temp_2);
                v_C_16 = vmlaq_s8(v_C_16, v_B_16_3, v_temp_3);
                vst1q_s8(&matC[i * 16 + j], v_C_16);
            }
        }
    }
}

void mmNeonMM_blocking_kernel_16x16_v0(int8_t* __restrict__ matA, int8_t* __restrict__ matB,
                                       int8_t* __restrict__ matC) {
    int i, j, k;
    for (i = 0; i < 16; i++) {
        for (k = 0; k <= 16; k++) {
            int8_t temp = matA[i * 16 + k];
            for (j = 0; j < 16; j++) {
                matC[i * 16 + j] += temp * matB[k * 16 + j];
            }
        }
    }


}
static int min(int a, int b) {
    return a<b?a:b;
}
void mmNeonMM_interchangeSIMD_v4(int8_t* __restrict__ A, int8_t* __restrict__ B, int8_t* __restrict__ C,
              int D_M, int D_N, int D_K) {
    const int block_size = 16;
    int i,j,k;
    for(i=0; i < D_M; i++) {
        for(j=0; j < D_N; j++) {
            C[i*D_N + j ] = 0;
        }
    }
    for (i = 0; i < D_M; i+=block_size) {
        for (j = 0; j < D_N; j+=block_size) {
            for (k=0; k < D_K; k+=block_size) {
                mmNeonMM_blocking_kernel_16x16_v0(&A[i*D_K+k], &B[k*D_N + j], &C[i*D_N+j]);
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
        mmNeonMM_interchangeSIMD_v4(matA, matB, matC, D_M, D_N, D_K);
    }
}