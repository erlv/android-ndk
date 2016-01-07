//
// Created by kunling on 12/26/15.
//

#include <stdint.h>
#include <arm_neon.h>
#include "gflops.h"

float test_f32_mac_Neon(float x, float y, uint64_t iterations){
    register float32x4_t r0,r1,r2,r3,r4,r5,r6,r7,r8,r9,rA,rB,rC,rD,rE,rF;

    //  Generate starting data.
    r0 = vdupq_n_f32(x);
    r1 = vdupq_n_f32(y);

    r8 = vdupq_n_f32(-0.0);

    r2 = vreinterpretq_f32_s32(veorq_s32(vreinterpretq_s32_f32(r0),vreinterpretq_s32_f32(r8)));
    r3 = vreinterpretq_f32_s32(vorrq_s32(vreinterpretq_s32_f32(r0),vreinterpretq_s32_f32(r8)));
    r4 = vreinterpretq_f32_s32(vandq_s32(vreinterpretq_s32_f32(r0),vreinterpretq_s32_f32(r8)));
    r5 = vmulq_f32(r1, vdupq_n_f32(0.37796447300922722721));
    r6 = vmulq_f32(r1, vdupq_n_f32(0.24253562503633297352));
    r7 = vmulq_f32(r1, vdupq_n_f32(4.1231056256176605498));
    r8 = vaddq_f32(r0, vdupq_n_f32(0.37796447300922722721));
    r9 = vaddq_f32(r1, vdupq_n_f32(0.24253562503633297352));
    rA = vsubq_f32(r0, vdupq_n_f32(4.1231056256176605498));
    rB = vsubq_f32(r1, vdupq_n_f32(4.1231056256176605498));

    rC = vdupq_n_f32(1.4142135623730950488);
    rD = vdupq_n_f32(1.7320508075688772935);
    rE = vdupq_n_f32(0.57735026918962576451);
    rF = vdupq_n_f32(0.70710678118654752440);

    int32x4_t MASK = vreinterpretq_s32_u32(vdupq_n_u32(0x800fffffu));
    int32x4_t vONE = vreinterpretq_s32_u32(vdupq_n_u32(1u));

    uint64_t c = 0;
    while (c < iterations){
        size_t i = 0;
        while (i < GFLOPS_INNER_ITERATION){
            //  Here's the meat - the part that really matters.
/*
 *  MUL + ADD + SUB
 */
            r0 = vmulq_f32(r0,rC);
            r1 = vaddq_f32(r1,rD);
            r2 = vmulq_f32(r2,rE);
            r3 = vsubq_f32(r3,rF);
            r4 = vmulq_f32(r4,rC);
            r5 = vaddq_f32(r5,rD);
            r6 = vmulq_f32(r6,rE);
            r7 = vsubq_f32(r7,rF);
            r8 = vmulq_f32(r8,rC);
            r9 = vaddq_f32(r9,rD);
            rA = vmulq_f32(rA,rE);
            rB = vsubq_f32(rB,rF);

            r0 = vaddq_f32(r0,rF);
            r1 = vmulq_f32(r1,rE);
            r2 = vsubq_f32(r2,rD);
            r3 = vmulq_f32(r3,rC);
            r4 = vaddq_f32(r4,rF);
            r5 = vmulq_f32(r5,rE);
            r6 = vsubq_f32(r6,rD);
            r7 = vmulq_f32(r7,rC);
            r8 = vaddq_f32(r8,rF);
            r9 = vmulq_f32(r9,rE);
            rA = vsubq_f32(rA,rD);
            rB = vmulq_f32(rB,rC);

            r0 = vmulq_f32(r0,rC);
            r1 = vaddq_f32(r1,rD);
            r2 = vmulq_f32(r2,rE);
            r3 = vsubq_f32(r3,rF);
            r4 = vmulq_f32(r4,rC);
            r5 = vaddq_f32(r5,rD);
            r6 = vmulq_f32(r6,rE);
            r7 = vsubq_f32(r7,rF);
            r8 = vmulq_f32(r8,rC);
            r9 = vaddq_f32(r9,rD);
            rA = vmulq_f32(rA,rE);
            rB = vsubq_f32(rB,rF);

            r0 = vaddq_f32(r0,rF);
            r1 = vmulq_f32(r1,rE);
            r2 = vsubq_f32(r2,rD);
            r3 = vmulq_f32(r3,rC);
            r4 = vaddq_f32(r4,rF);
            r5 = vmulq_f32(r5,rE);
            r6 = vsubq_f32(r6,rD);
            r7 = vmulq_f32(r7,rC);
            r8 = vaddq_f32(r8,rF);
            r9 = vmulq_f32(r9,rE);
            rA = vsubq_f32(rA,rD);
            rB = vmulq_f32(rB,rC);

            i++;
        }

        //  Need to renormalize to prevent denormal/overflow.

        r0 = vreinterpretq_f32_s32(vandq_s32(vreinterpretq_s32_f32(r0),MASK));
        r1 = vreinterpretq_f32_s32(vandq_s32(vreinterpretq_s32_f32((r1)),MASK));
        r2 = vreinterpretq_f32_s32(vandq_s32(vreinterpretq_s32_f32((r2)),MASK));
        r3 = vreinterpretq_f32_s32(vandq_s32(vreinterpretq_s32_f32((r3)),MASK));
        r4 = vreinterpretq_f32_s32(vandq_s32(vreinterpretq_s32_f32((r4)),MASK));
        r5 = vreinterpretq_f32_s32(vandq_s32(vreinterpretq_s32_f32((r5)),MASK));
        r6 = vreinterpretq_f32_s32(vandq_s32(vreinterpretq_s32_f32((r6)),MASK));
        r7 = vreinterpretq_f32_s32(vandq_s32(vreinterpretq_s32_f32((r7)),MASK));
        r8 = vreinterpretq_f32_s32(vandq_s32(vreinterpretq_s32_f32((r8)),MASK));
        r9 = vreinterpretq_f32_s32(vandq_s32(vreinterpretq_s32_f32((r9)),MASK));
        rA = vreinterpretq_f32_s32(vandq_s32(vreinterpretq_s32_f32((rA)),MASK));
        rB = vreinterpretq_f32_s32(vandq_s32(vreinterpretq_s32_f32((rB)),MASK));
        r0 = vreinterpretq_f32_s32(vorrq_s32(vreinterpretq_s32_f32((r0)),vONE));
        r1 = vreinterpretq_f32_s32(vorrq_s32(vreinterpretq_s32_f32((r1)),vONE));
        r2 = vreinterpretq_f32_s32(vorrq_s32(vreinterpretq_s32_f32((r2)),vONE));
        r3 = vreinterpretq_f32_s32(vorrq_s32(vreinterpretq_s32_f32((r3)),vONE));
        r4 = vreinterpretq_f32_s32(vorrq_s32(vreinterpretq_s32_f32((r4)),vONE));
        r5 = vreinterpretq_f32_s32(vorrq_s32(vreinterpretq_s32_f32((r5)),vONE));
        r6 = vreinterpretq_f32_s32(vorrq_s32(vreinterpretq_s32_f32((r6)),vONE));
        r7 = vreinterpretq_f32_s32(vorrq_s32(vreinterpretq_s32_f32((r7)),vONE));
        r8 = vreinterpretq_f32_s32(vorrq_s32(vreinterpretq_s32_f32((r8)),vONE));
        r9 = vreinterpretq_f32_s32(vorrq_s32(vreinterpretq_s32_f32((r9)),vONE));
        rA = vreinterpretq_f32_s32(vorrq_s32(vreinterpretq_s32_f32((rA)),vONE));
        rB = vreinterpretq_f32_s32(vorrq_s32(vreinterpretq_s32_f32((rB)),vONE));
        c++;
    }

    r0 = vaddq_f32(r0,r1);
    r2 = vaddq_f32(r2,r3);
    r4 = vaddq_f32(r4,r5);
    r6 = vaddq_f32(r6,r7);
    r8 = vaddq_f32(r8,r9);
    rA = vaddq_f32(rA,rB);

    r0 = vaddq_f32(r0,r2);
    r4 = vaddq_f32(r4,r6);
    r8 = vaddq_f32(r8,rA);

    r0 = vaddq_f32(r0,r4);
    r0 = vaddq_f32(r0,r8);

    //  Prevent Dead Code Elimination
    float out = 0;
    float32x4_t temp = r0;
    out += ((float*)&temp)[0];
    out += ((float*)&temp)[1];
    out += ((float*)&temp)[3];
    out += ((float*)&temp)[4];
    return out;
}

int32_t test_s32_mac_Neon(int32_t x, int32_t y, uint64_t iterations){
    register int32x4_t r0,r1,r2,r3,r4,r5,r6,r7,r8,r9,rA,rB,rC,rD,rE,rF;

    //  Generate starting data.
    r0 = vdupq_n_s32((int32_t)x);
    r1 = vdupq_n_s32((int32_t)y);

    r8 = vdupq_n_s32(0);

    r2 = (veorq_s32((r0),(r8)));
    r3 = (vorrq_s32((r0),(r8)));
    r4 = (vandq_s32((r0),(r8)));
    r5 = vmulq_s32(r1, vdupq_n_s32(3));
    r6 = vmulq_s32(r1, vdupq_n_s32(2));
    r7 = vmulq_s32(r1, vdupq_n_s32(1));
    r8 = vaddq_s32(r0, vdupq_n_s32(3));
    r9 = vaddq_s32(r1, vdupq_n_s32(2));
    rA = vsubq_s32(r0, vdupq_n_s32(4));
    rB = vsubq_s32(r1, vdupq_n_s32(4));

    rC = vdupq_n_s32(4);
    rD = vdupq_n_s32(7);
    rE = vdupq_n_s32(5);
    rF = vdupq_n_s32(7);

    int32x4_t MASK = vreinterpretq_s32_u32(vdupq_n_u32(0x800fffffu));
    int32x4_t vONE = vreinterpretq_s32_u32(vdupq_n_u32(1u));

    uint64_t c = 0;
    while (c < iterations){
        size_t i = 0;
        while (i < GFLOPS_INNER_ITERATION){
            //  Here's the meat - the part that really matters.

            r0 = vmulq_s32(r0,rC);
            r1 = vsubq_s32(r1,rD);
            r2 = vmulq_s32(r2,rE);
            r3 = vsubq_s32(r3,rF);
            r4 = vmulq_s32(r4,rC);
            r5 = vsubq_s32(r5,rD);
            r6 = vmulq_s32(r6,rE);
            r7 = vsubq_s32(r7,rF);
            r8 = vmulq_s32(r8,rC);
            r9 = vsubq_s32(r9,rD);
            rA = vmulq_s32(rA,rE);
            rB = vsubq_s32(rB,rF);

            r0 = vsubq_s32(r0,rF);
            r1 = vmulq_s32(r1,rE);
            r2 = vsubq_s32(r2,rD);
            r3 = vmulq_s32(r3,rC);
            r4 = vsubq_s32(r4,rF);
            r5 = vmulq_s32(r5,rE);
            r6 = vsubq_s32(r6,rD);
            r7 = vmulq_s32(r7,rC);
            r8 = vsubq_s32(r8,rF);
            r9 = vmulq_s32(r9,rE);
            rA = vsubq_s32(rA,rD);
            rB = vmulq_s32(rB,rC);

            r0 = vmulq_s32(r0,rC);
            r1 = vsubq_s32(r1,rD);
            r2 = vmulq_s32(r2,rE);
            r3 = vsubq_s32(r3,rF);
            r4 = vmulq_s32(r4,rC);
            r5 = vsubq_s32(r5,rD);
            r6 = vmulq_s32(r6,rE);
            r7 = vsubq_s32(r7,rF);
            r8 = vmulq_s32(r8,rC);
            r9 = vsubq_s32(r9,rD);
            rA = vmulq_s32(rA,rE);
            rB = vsubq_s32(rB,rF);

            r0 = vsubq_s32(r0,rF);
            r1 = vmulq_s32(r1,rE);
            r2 = vsubq_s32(r2,rD);
            r3 = vmulq_s32(r3,rC);
            r4 = vsubq_s32(r4,rF);
            r5 = vmulq_s32(r5,rE);
            r6 = vsubq_s32(r6,rD);
            r7 = vmulq_s32(r7,rC);
            r8 = vsubq_s32(r8,rF);
            r9 = vmulq_s32(r9,rE);
            rA = vsubq_s32(rA,rD);
            rB = vmulq_s32(rB,rC);
            i++;
        }

        //  Need to renormalize to prevent denormal/overflow.

        r0 = (vandq_s32((r0),MASK));
        r1 = (vandq_s32(((r1)),MASK));
        r2 = (vandq_s32(((r2)),MASK));
        r3 = (vandq_s32(((r3)),MASK));
        r4 = (vandq_s32(((r4)),MASK));
        r5 = (vandq_s32(((r5)),MASK));
        r6 = (vandq_s32(((r6)),MASK));
        r7 = (vandq_s32(((r7)),MASK));
        r8 = (vandq_s32(((r8)),MASK));
        r9 = (vandq_s32(((r9)),MASK));
        rA = (vandq_s32(((rA)),MASK));
        rB = (vandq_s32(((rB)),MASK));
        r0 = (vorrq_s32(((r0)),vONE));
        r1 = (vorrq_s32(((r1)),vONE));
        r2 = (vorrq_s32(((r2)),vONE));
        r3 = (vorrq_s32(((r3)),vONE));
        r4 = (vorrq_s32(((r4)),vONE));
        r5 = (vorrq_s32(((r5)),vONE));
        r6 = (vorrq_s32(((r6)),vONE));
        r7 = (vorrq_s32(((r7)),vONE));
        r8 = (vorrq_s32(((r8)),vONE));
        r9 = (vorrq_s32(((r9)),vONE));
        rA = (vorrq_s32(((rA)),vONE));
        rB = (vorrq_s32(((rB)),vONE));
        c++;
    }

    r0 = vaddq_s32(r0,r1);
    r2 = vaddq_s32(r2,r3);
    r4 = vaddq_s32(r4,r5);
    r6 = vaddq_s32(r6,r7);
    r8 = vaddq_s32(r8,r9);
    rA = vaddq_s32(rA,rB);

    r0 = vaddq_s32(r0,r2);
    r4 = vaddq_s32(r4,r6);
    r8 = vaddq_s32(r8,rA);

    r0 = vaddq_s32(r0,r4);
    r0 = vaddq_s32(r0,r8);

    //  Prevent Dead Code Elimination
    int32_t out_s32 = 0;
    int32x4_t temp = r0;
    out_s32 += ((int32_t*)&temp)[0];
    out_s32 += ((int32_t*)&temp)[1];
    out_s32 += ((int32_t*)&temp)[3];
    out_s32 += ((int32_t*)&temp)[4];
    return out_s32;
}

float peakGFLOPS(uint64_t iterations) {
    return test_f32_mac_Neon(1.1,2.1,iterations);
}

int32_t peakInt8GOps(uint64_t iterations) {
    return test_s32_mac_Neon(1, 2, iterations);
}