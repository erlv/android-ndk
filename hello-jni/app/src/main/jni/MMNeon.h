//
// Created by kunling on 12/25/15.
//

#ifndef HELLO_JNI_MMNEON_H
#define HELLO_JNI_MMNEON_H

#include <stdint.h>

#define BLOCKSIZE 128


void mmNeon(int8_t* __restrict__ A, int8_t* __restrict__ B, int8_t* __restrict__ C,
              int D_M, int D_N, int D_K);
#endif //HELLO_JNI_MMNEON_H
