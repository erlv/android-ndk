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
#include <arm_neon.h>
#include <jni.h>
#include <stdlib.h>
#include <stdbool.h>
#include <stdint.h>

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



#define N  256
#define MAX 6
int16_t A[N];
int16_t B[N];
int16_t C_1[N];
int16_t C_2[N];

void add_c() {
    int i=0;
    for (; i < N; i++) {
        C_1[i] = A[i] + B[i];
    }
}

void add_neon_c() {
    int i=0;
    for (; i < N; i +=8) {
        int16x8_t v_a = vld1q_s16(&A[i]);
        int16x8_t v_b = vld1q_s16(&B[i]);
        int16x8_t v_c = vaddq_s16(v_a, v_b);
        vst1q_s16(&C_2[i], v_c);
    }
}


bool compare_arr() {
    int i=0;
    for(; i < N; i++) {
        if (C_1[i] != C_2[i]) {
            printf("%d: %d != %d\n", i, C_1[i], C_2[i]);
            return false;
        }
    }
    return true;
}

char* fake_main() {
    int i=0;
    srand(0);
    for (; i< N; i++) {
        A[i] = rand()/MAX;
        B[i] = rand()/MAX;
        C_1[i] = rand()/MAX;
        C_2[i] = rand()/MAX;
    }
    add_c();
    add_neon_c();
    bool is_pass = compare_arr();
    char* buf = malloc(JSTRING_BUF_SIZE*sizeof(char));
    memset(buf, 0, JSTRING_BUF_SIZE*sizeof(char));
    sprintf(buf, "%s %s: %s\n Neon minitest:%s\n", __DATE__, __TIME__, "Hello from JNI !Compiled with ABI "
    ABI "." , is_pass?"PASS":"FAIL");
    return buf;
}

int main() {
    char* buf = fake_main();
    printf("%s",buf);
    free(buf);
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
    char* buf = fake_main();
    return (*env)->NewStringUTF(env, buf);
}
