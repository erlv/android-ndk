/*
 * Copyright (C) 2009 The Android Open Source Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */
#include <string.h>
#include <stdio.h>
#include <jni.h>
#include <stdlib.h>
#include <stdbool.h>
#include <sys/time.h>
#include "MMScalar.h"
#include "MMNeon.h"

#define JSTRING_BUF_SIZE 4096
#if defined(__arm__)
#if defined(__ARM_ARCH_7A__)
#if defined(__ARM_NEON__)
#if defined(__ARM_PCS_VFP)
#define ABI "armeabi-v7a/NEON (hard-float)"
#else
#define ABI "armeabi-v7a/NEON"
#endif
#else
#if defined(__ARM_PCS_VFP)
  #define ABI "armeabi-v7a (hard-float)"
#else
  #define ABI "armeabi-v7a"
#endif
#endif
#else
#define ABI "armeabi"
#endif
#else
#error currently the code is only for ARMV7 + NEON
#endif

#define MMREPEAT 1000

char* jni_buf;
char* cur_jni_ptr;

// Optimize for:
// S1. A (512, 2048), B (2048, 1) = C ( 512, 1)
// S2. A (8000, 640), B (640, 1) = C ( 8000， 1）
// M1. A (512, 2048), B (2048, 128) = C ( 512, 128)
// M2. A (8000, 640), B (640, 128) = C ( 8000， 128）
void initRandomArray(int8_t * a, size_t sz) {
    int i =0;
    for(; i <sz; i++) {
        int8_t r3 =  (int8_t) (rand()); // % 3;
        a[i] = r3;
    }
}

bool compareRes(int8_t * res, int8_t* exp, size_t sz) {
    int i = 0 ;
    for (; i < sz; i++) {
        if (res[i] != exp[i]) {
            int char_cnt = sprintf(cur_jni_ptr, "%d:%d != %d\n",
                i, res[i], exp[i]);
            cur_jni_ptr += char_cnt;
            return false;
        }
    }
    if (i  == sz)
        return true;
    else
        return false;
}

static long long currentTimeInMilliseconds()
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return ((tv.tv_sec * 1000) + (tv.tv_usec / 1000));
}

int fake_main() {
    jni_buf = malloc(JSTRING_BUF_SIZE*sizeof(char));
    memset(jni_buf, 0, JSTRING_BUF_SIZE*sizeof(char));
    cur_jni_ptr = jni_buf;
    srand(0);
    int char_cnt;

    // malloc the memory for 3 arries
    int A_R, A_C, B_R, B_C, C_R, C_C;
    // S1
    const char* cur_str = "S1"; C_R = A_R = 512; B_R = A_C = 2048; C_C = B_C = 1;

#if 0
    // S2
    const char* cur_str = "S2"; A_R = 8000; A_C = 640; B_R = 640; B_C = 1; C_R = 8000; C_C = 1;
    // M1
    const char* cur_str = "M1"; A_R = 512; A_C = 2048; B_R = 2048; B_C = 128; C_R = 512; C_C = 128;
    // M2
    const char* cur_str = "M2"; A_R = 8000; A_C = 640; B_R = 640; B_C = 128; C_R = 8000; C_C = 128;
    // Random
#endif

    // 0. Output test information:
    char_cnt = sprintf(cur_jni_ptr, " Test Pattern: dim(M)=(%d, %d), dim(x)=(%d, %d), "
                                    "result=(%d,%d)\n", A_R, A_C, B_R, B_C, C_R, C_C);
    cur_jni_ptr += char_cnt;

    // 1. Init and malloc the array
    int8_t* matA = malloc(A_R*A_C*sizeof(char));
    int8_t* matB = malloc(B_R*B_C*sizeof(char));
    int8_t* matC = malloc(C_R*C_C*sizeof(char));
    int8_t* verifyC = malloc(C_R*C_C*sizeof(char));
    initRandomArray(matA, A_R*A_C);
    initRandomArray(matB, B_R*B_C);
    memset(matC, 0, C_R*C_C);

    // 2. Prepare the verifyC result
    mmScalar(matA, matB, verifyC, A_R, C_C, A_C);

    // 3. Prepare the 1st Neon Optimized matC result
    mmNeon(matA, matB, matC, A_R, C_C, A_C);

    // 4. Verify the result
    if (!compareRes(matC, verifyC, C_R*C_C)) {
        char_cnt = sprintf(cur_jni_ptr, " Verification: FAILED\n");
        cur_jni_ptr += char_cnt;
        return 0;
    } else {
        char_cnt = sprintf(cur_jni_ptr, "Verification: PASSED\n");
        cur_jni_ptr += char_cnt;
    }

    long long prev_curms, after_curms;
    int i;
    // 5. Performance test for sequential MM
    // TODO: add timer for sequentail MM
    prev_curms = currentTimeInMilliseconds();
    // run sequential MM MMREPEAT Times
    for(  i=0; i < MMREPEAT; i++) {
        mmScalar(matA, matB, verifyC, A_R, C_C, A_C);
    }
    after_curms = currentTimeInMilliseconds();
    float gops = (2.0 * MMREPEAT * A_R * C_C * A_C) / ((float)((after_curms - prev_curms) * 1000000));
    char_cnt = sprintf(cur_jni_ptr, "%d x SequentialMM runtime: %lldms, %fGOPS\n", MMREPEAT,
                           after_curms - prev_curms, gops);
    cur_jni_ptr += char_cnt;

    // 6. Performance test for Neon Optimized MM
    // TODO: add tmer for Neon optimized MM
    prev_curms = currentTimeInMilliseconds();
    // run sequential MM MMREPEAT Times
    for(  i=0; i < MMREPEAT; i++) {
        mmNeon(matA, matB, verifyC, A_R, C_C, A_C);
    }
    after_curms = currentTimeInMilliseconds();
    gops = (2.0 * MMREPEAT * A_R * C_C * A_C) / ((float)((after_curms - prev_curms) * 1000000));
    char_cnt = sprintf(cur_jni_ptr, "%d x NeonMM runtime: %lldms, %fGOPS\n", MMREPEAT,
                           after_curms - prev_curms, gops);
    cur_jni_ptr += char_cnt;
    return 0;
}

int main() {
    fake_main();
    printf("%s",jni_buf);
    free(jni_buf);
    return 0;
}

/* This is a trivial JNI example where we use a native method
 * to return a new VM String. See the corresponding Java source
 * file located at:
 *
 *   apps/samples/hello-jni/project/src/com/example/hellojni/HelloJni.java
 */
jstring
Java_com_example_hellojni_HelloJni_stringFromJNI( JNIEnv* env,
                                                  jobject thiz )
{
    fake_main();
    return (*env)->NewStringUTF(env, jni_buf);
}