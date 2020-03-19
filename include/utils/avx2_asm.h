#pragma once

#ifdef __SSE__
#include <typeinfo>
#include <stdio.h>
#include <stdint.h>
#include <mmintrin.h>
#include <xmmintrin.h>
#include <pmmintrin.h>
#include <emmintrin.h>
#include <smmintrin.h>
#include <immintrin.h>

#include "rsearch_type.h"
namespace rsearch{
// ============== int8 =============== 


/*
*   for i = 0 to 3 do
*       for j = 0 to 1 do
*           c[j * ldc + i] = a[i] * b[j] + offset_ptr[i]
*/
template<int K>
void dot_4x2(const int8_t *a, const int8_t *b, const int * offset_ptr, int *c, int ldc){
    __m256i acc[8];
    __m256i aval[4], bval[2];
    __m256i one = _mm256_set1_epi16(1);
    for(int i=0; i<8; i++)
        acc[i] = _mm256_setzero_si256();
    for (int k = 0; k < K; k+=32) {
        aval[0] = _mm256_loadu_si256((__m256i *)(a));
        _mm_prefetch(a + 128, _MM_HINT_T0);
        aval[1] = _mm256_loadu_si256((__m256i *)(a+K*1));
        _mm_prefetch(a + 128 + K, _MM_HINT_T0);
        aval[2] = _mm256_loadu_si256((__m256i *)(a+K*2));
        _mm_prefetch(a + 128 + K * 2, _MM_HINT_T0);
        aval[3] = _mm256_loadu_si256((__m256i *)(a+K*3));
        _mm_prefetch(a + 128 + K * 3, _MM_HINT_T0);
        bval[0] =    _mm256_loadu_si256((__m256i *)(b));
        bval[1] =    _mm256_loadu_si256((__m256i *)(b + K));

        __m256i ymm[4];
        ymm[0]  = _mm256_maddubs_epi16(bval[0], aval[0]);
        ymm[1]  = _mm256_maddubs_epi16(bval[0], aval[1]);
        ymm[2]  = _mm256_maddubs_epi16(bval[0], aval[2]);
        ymm[3]  = _mm256_maddubs_epi16(bval[0], aval[3]);

        ymm[0]  = _mm256_madd_epi16(ymm[0], one);
        ymm[1]  = _mm256_madd_epi16(ymm[1], one);
        ymm[2]  = _mm256_madd_epi16(ymm[2], one);
        ymm[3]  = _mm256_madd_epi16(ymm[3], one);

        acc[0]  = _mm256_add_epi32(ymm[0], acc[0]);
        acc[1]  = _mm256_add_epi32(ymm[1], acc[1]);
        acc[2]  = _mm256_add_epi32(ymm[2], acc[2]);
        acc[3]  = _mm256_add_epi32(ymm[3], acc[3]);
 
        ymm[0]  = _mm256_maddubs_epi16(bval[1], aval[0]);
        ymm[1]  = _mm256_maddubs_epi16(bval[1], aval[1]);
        ymm[2]  = _mm256_maddubs_epi16(bval[1], aval[2]);
        ymm[3]  = _mm256_maddubs_epi16(bval[1], aval[3]);

        ymm[0]  = _mm256_madd_epi16(ymm[0], one);
        ymm[1]  = _mm256_madd_epi16(ymm[1], one);
        ymm[2]  = _mm256_madd_epi16(ymm[2], one);
        ymm[3]  = _mm256_madd_epi16(ymm[3], one);

        acc[4]  = _mm256_add_epi32(ymm[0], acc[4]);
        acc[5]  = _mm256_add_epi32(ymm[1], acc[5]);
        acc[6]  = _mm256_add_epi32(ymm[2], acc[6]);
        acc[7]  = _mm256_add_epi32(ymm[3], acc[7]);
        a += 32;
        b += 32;
    }
    __m128i xmm[8];
    __m128i offset;
    
    xmm[0] = _mm256_extracti128_si256(acc[0], 0);
    xmm[1] = _mm256_extracti128_si256(acc[0], 1);
    xmm[2] = _mm256_extracti128_si256(acc[1], 0);
    xmm[3] = _mm256_extracti128_si256(acc[1], 1);

    xmm[4] = _mm256_extracti128_si256(acc[2], 0);
    xmm[5] = _mm256_extracti128_si256(acc[2], 1);
    xmm[6] = _mm256_extracti128_si256(acc[3], 0);
    xmm[7] = _mm256_extracti128_si256(acc[3], 1);

    xmm[0] = _mm_add_epi32(xmm[0], xmm[1]);
    xmm[2] = _mm_add_epi32(xmm[2], xmm[3]);
    xmm[4] = _mm_add_epi32(xmm[4], xmm[5]);
    xmm[6] = _mm_add_epi32(xmm[6], xmm[7]);

    xmm[0] = _mm_hadd_epi32(xmm[0], xmm[2]);
    xmm[4] = _mm_hadd_epi32(xmm[4], xmm[6]);
    xmm[0] = _mm_hadd_epi32(xmm[0], xmm[4]);
 
    offset = _mm_loadu_si128((__m128i *)offset_ptr);
    xmm[0] = _mm_add_epi32(xmm[0], offset);
    _mm_storeu_si128((__m128i *)c, xmm[0]);
 
    xmm[0] = _mm256_extracti128_si256(acc[4], 0);
    xmm[1] = _mm256_extracti128_si256(acc[4], 1);
    xmm[2] = _mm256_extracti128_si256(acc[5], 0);
    xmm[3] = _mm256_extracti128_si256(acc[5], 1);

    xmm[4] = _mm256_extracti128_si256(acc[6], 0);
    xmm[5] = _mm256_extracti128_si256(acc[6], 1);
    xmm[6] = _mm256_extracti128_si256(acc[7], 0);
    xmm[7] = _mm256_extracti128_si256(acc[7], 1);

    xmm[0] = _mm_add_epi32(xmm[0], xmm[1]);
    xmm[2] = _mm_add_epi32(xmm[2], xmm[3]);
    xmm[4] = _mm_add_epi32(xmm[4], xmm[5]);
    xmm[6] = _mm_add_epi32(xmm[6], xmm[7]);

    xmm[0] = _mm_hadd_epi32(xmm[0], xmm[2]);
    xmm[4] = _mm_hadd_epi32(xmm[4], xmm[6]);
    xmm[0] = _mm_hadd_epi32(xmm[0], xmm[4]);
 
    xmm[0] = _mm_add_epi32(xmm[0], offset);
    _mm_storeu_si128((__m128i *)(c + ldc), xmm[0]);

}


template<int K>
void dot_1x1(const int8_t *a, const int8_t *b, const int * offset_ptr, int *c){
    __m256i acc;
    __m256i aval, bval;
    __m256i ymm;
    __m256i one = _mm256_set1_epi16(1);

    acc = _mm256_setzero_si256();

    for (int k = 0; k < K; k+=32) {
        aval = _mm256_loadu_si256((__m256i *)(a));
        bval = _mm256_loadu_si256((__m256i *)(b));
        _mm_prefetch(a + 128, _MM_HINT_T0);

        ymm  = _mm256_maddubs_epi16(bval, aval);
        ymm  = _mm256_madd_epi16(ymm, one);
        acc  = _mm256_add_epi32(ymm, acc);

        a += 32;
        b += 32;
    }

    __m128i xmm[2];
    
    xmm[0] = _mm256_extracti128_si256(acc, 0);
    xmm[1] = _mm256_extracti128_si256(acc, 1);

    xmm[0] = _mm_add_epi32(xmm[0], xmm[1]);

    xmm[0] = _mm_hadd_epi32(xmm[0], xmm[0]);
    xmm[0] = _mm_hadd_epi32(xmm[0], xmm[0]);
 
    c[0] =  _mm_extract_epi32(xmm[0], 0) + offset_ptr[0];
}


template<int K>
void dot_4x1(const int8_t *a, const int8_t *b, const int * offset_ptr, int *c){
    __m256i acc[4];
    __m256i aval[4], bval;
    __m256i ymm[4];
    __m256i one = _mm256_set1_epi16(1);
    for(int i=0; i<4; i++)
        acc[i] = _mm256_setzero_si256();
    for (int k = 0; k < K; k+=32) {
        aval[0] = _mm256_loadu_si256((__m256i *)(a));
        _mm_prefetch(a + 128, _MM_HINT_T0);
        aval[1] = _mm256_loadu_si256((__m256i *)(a+K*1));
        _mm_prefetch(a + 128 + K, _MM_HINT_T0);
        aval[2] = _mm256_loadu_si256((__m256i *)(a+K*2));
        _mm_prefetch(a + 128 + K * 2, _MM_HINT_T0);
        aval[3] = _mm256_loadu_si256((__m256i *)(a+K*3));
        _mm_prefetch(a + 128 + K * 3, _MM_HINT_T0);
        bval =    _mm256_loadu_si256((__m256i *)(b));
        ymm[0]  = _mm256_maddubs_epi16(bval, aval[0]);
        ymm[1]  = _mm256_maddubs_epi16(bval, aval[1]);
        ymm[2]  = _mm256_maddubs_epi16(bval, aval[2]);
        ymm[3]  = _mm256_maddubs_epi16(bval, aval[3]);

        ymm[0]  = _mm256_madd_epi16(ymm[0], one);
        ymm[1]  = _mm256_madd_epi16(ymm[1], one);
        ymm[2]  = _mm256_madd_epi16(ymm[2], one);
        ymm[3]  = _mm256_madd_epi16(ymm[3], one);

        acc[0]  = _mm256_add_epi32(ymm[0], acc[0]);
        acc[1]  = _mm256_add_epi32(ymm[1], acc[1]);
        acc[2]  = _mm256_add_epi32(ymm[2], acc[2]);
        acc[3]  = _mm256_add_epi32(ymm[3], acc[3]);
        a += 32;
        b += 32;
    }
    __m128i xmm[8];
    __m128i offset;
    
    xmm[0] = _mm256_extracti128_si256(acc[0], 0);
    xmm[1] = _mm256_extracti128_si256(acc[0], 1);
    xmm[2] = _mm256_extracti128_si256(acc[1], 0);
    xmm[3] = _mm256_extracti128_si256(acc[1], 1);

    xmm[4] = _mm256_extracti128_si256(acc[2], 0);
    xmm[5] = _mm256_extracti128_si256(acc[2], 1);
    xmm[6] = _mm256_extracti128_si256(acc[3], 0);
    xmm[7] = _mm256_extracti128_si256(acc[3], 1);

    xmm[0] = _mm_add_epi32(xmm[0], xmm[1]);
    xmm[2] = _mm_add_epi32(xmm[2], xmm[3]);
    xmm[4] = _mm_add_epi32(xmm[4], xmm[5]);
    xmm[6] = _mm_add_epi32(xmm[6], xmm[7]);

    xmm[0] = _mm_hadd_epi32(xmm[0], xmm[2]);
    xmm[4] = _mm_hadd_epi32(xmm[4], xmm[6]);
    xmm[0] = _mm_hadd_epi32(xmm[0], xmm[4]);
 
    offset = _mm_loadu_si128((__m128i *)offset_ptr);
    xmm[0] = _mm_add_epi32(xmm[0], offset);
    _mm_storeu_si128((__m128i *)c, xmm[0]);
}


void dot_nt_4x2(const int8_t *a, const int8_t *b, const int * offset_ptr, const int K, int *c, int ldc){
    __m256i acc[8];
    __m256i aval[4], bval[2];
    __m256i one = _mm256_set1_epi16(1);
    for(int i=0; i<8; i++)
        acc[i] = _mm256_setzero_si256();
    for (int k = 0; k < K; k+=32) {
        aval[0] = _mm256_loadu_si256((__m256i *)(a));
        _mm_prefetch(a + 128, _MM_HINT_T0);
        aval[1] = _mm256_loadu_si256((__m256i *)(a+K*1));
        _mm_prefetch(a + 128 + K, _MM_HINT_T0);
        aval[2] = _mm256_loadu_si256((__m256i *)(a+K*2));
        _mm_prefetch(a + 128 + K * 2, _MM_HINT_T0);
        aval[3] = _mm256_loadu_si256((__m256i *)(a+K*3));
        _mm_prefetch(a + 128 + K * 3, _MM_HINT_T0);
        bval[0] =    _mm256_loadu_si256((__m256i *)(b));
        bval[1] =    _mm256_loadu_si256((__m256i *)(b + K));

        __m256i ymm[4];
        ymm[0]  = _mm256_maddubs_epi16(bval[0], aval[0]);
        ymm[1]  = _mm256_maddubs_epi16(bval[0], aval[1]);
        ymm[2]  = _mm256_maddubs_epi16(bval[0], aval[2]);
        ymm[3]  = _mm256_maddubs_epi16(bval[0], aval[3]);

        ymm[0]  = _mm256_madd_epi16(ymm[0], one);
        ymm[1]  = _mm256_madd_epi16(ymm[1], one);
        ymm[2]  = _mm256_madd_epi16(ymm[2], one);
        ymm[3]  = _mm256_madd_epi16(ymm[3], one);

        acc[0]  = _mm256_add_epi32(ymm[0], acc[0]);
        acc[1]  = _mm256_add_epi32(ymm[1], acc[1]);
        acc[2]  = _mm256_add_epi32(ymm[2], acc[2]);
        acc[3]  = _mm256_add_epi32(ymm[3], acc[3]);
 
        ymm[0]  = _mm256_maddubs_epi16(bval[1], aval[0]);
        ymm[1]  = _mm256_maddubs_epi16(bval[1], aval[1]);
        ymm[2]  = _mm256_maddubs_epi16(bval[1], aval[2]);
        ymm[3]  = _mm256_maddubs_epi16(bval[1], aval[3]);

        ymm[0]  = _mm256_madd_epi16(ymm[0], one);
        ymm[1]  = _mm256_madd_epi16(ymm[1], one);
        ymm[2]  = _mm256_madd_epi16(ymm[2], one);
        ymm[3]  = _mm256_madd_epi16(ymm[3], one);

        acc[4]  = _mm256_add_epi32(ymm[0], acc[4]);
        acc[5]  = _mm256_add_epi32(ymm[1], acc[5]);
        acc[6]  = _mm256_add_epi32(ymm[2], acc[6]);
        acc[7]  = _mm256_add_epi32(ymm[3], acc[7]);
        a += 32;
        b += 32;
    }
    __m128i xmm[8];
    __m128i offset;
    
    xmm[0] = _mm256_extracti128_si256(acc[0], 0);
    xmm[1] = _mm256_extracti128_si256(acc[0], 1);
    xmm[2] = _mm256_extracti128_si256(acc[1], 0);
    xmm[3] = _mm256_extracti128_si256(acc[1], 1);

    xmm[4] = _mm256_extracti128_si256(acc[2], 0);
    xmm[5] = _mm256_extracti128_si256(acc[2], 1);
    xmm[6] = _mm256_extracti128_si256(acc[3], 0);
    xmm[7] = _mm256_extracti128_si256(acc[3], 1);

    xmm[0] = _mm_add_epi32(xmm[0], xmm[1]);
    xmm[2] = _mm_add_epi32(xmm[2], xmm[3]);
    xmm[4] = _mm_add_epi32(xmm[4], xmm[5]);
    xmm[6] = _mm_add_epi32(xmm[6], xmm[7]);

    xmm[0] = _mm_hadd_epi32(xmm[0], xmm[2]);
    xmm[4] = _mm_hadd_epi32(xmm[4], xmm[6]);
    xmm[0] = _mm_hadd_epi32(xmm[0], xmm[4]);
 
    offset = _mm_loadu_si128((__m128i *)offset_ptr);
    xmm[0] = _mm_add_epi32(xmm[0], offset);
    _mm_storeu_si128((__m128i *)c, xmm[0]);
 
    xmm[0] = _mm256_extracti128_si256(acc[4], 0);
    xmm[1] = _mm256_extracti128_si256(acc[4], 1);
    xmm[2] = _mm256_extracti128_si256(acc[5], 0);
    xmm[3] = _mm256_extracti128_si256(acc[5], 1);

    xmm[4] = _mm256_extracti128_si256(acc[6], 0);
    xmm[5] = _mm256_extracti128_si256(acc[6], 1);
    xmm[6] = _mm256_extracti128_si256(acc[7], 0);
    xmm[7] = _mm256_extracti128_si256(acc[7], 1);

    xmm[0] = _mm_add_epi32(xmm[0], xmm[1]);
    xmm[2] = _mm_add_epi32(xmm[2], xmm[3]);
    xmm[4] = _mm_add_epi32(xmm[4], xmm[5]);
    xmm[6] = _mm_add_epi32(xmm[6], xmm[7]);

    xmm[0] = _mm_hadd_epi32(xmm[0], xmm[2]);
    xmm[4] = _mm_hadd_epi32(xmm[4], xmm[6]);
    xmm[0] = _mm_hadd_epi32(xmm[0], xmm[4]);
 
    xmm[0] = _mm_add_epi32(xmm[0], offset);
    _mm_storeu_si128((__m128i *)(c + ldc), xmm[0]);

}

void dot_nt_1x1(const int8_t *a, const int8_t *b, const int * offset_ptr, const int K, int *c){
    __m256i acc;
    __m256i aval, bval;
    __m256i ymm;
    __m256i one = _mm256_set1_epi16(1);

    acc = _mm256_setzero_si256();

    for (int k = 0; k < K; k+=32) {
        aval = _mm256_loadu_si256((__m256i *)(a));
        bval = _mm256_loadu_si256((__m256i *)(b));
        _mm_prefetch(a + 128, _MM_HINT_T0);

        ymm  = _mm256_maddubs_epi16(bval, aval);
        ymm  = _mm256_madd_epi16(ymm, one);
        acc  = _mm256_add_epi32(ymm, acc);

        a += 32;
        b += 32;
    }

    __m128i xmm[2];
    
    xmm[0] = _mm256_extracti128_si256(acc, 0);
    xmm[1] = _mm256_extracti128_si256(acc, 1);

    xmm[0] = _mm_add_epi32(xmm[0], xmm[1]);

    xmm[0] = _mm_hadd_epi32(xmm[0], xmm[0]);
    xmm[0] = _mm_hadd_epi32(xmm[0], xmm[0]);
 
    c[0] =  _mm_extract_epi32(xmm[0], 0) + offset_ptr[0];
}

void dot_nt_4x1(const int8_t *a, const int8_t *b, const int * offset_ptr, const int K, int *c){
    __m256i acc[4];
    __m256i aval[4], bval;
    __m256i ymm[4];
    __m256i one = _mm256_set1_epi16(1);
    for(int i=0; i<4; i++)
        acc[i] = _mm256_setzero_si256();
    for (int k = 0; k < K; k+=32) {
        aval[0] = _mm256_loadu_si256((__m256i *)(a));
        _mm_prefetch(a + 128, _MM_HINT_T0);
        aval[1] = _mm256_loadu_si256((__m256i *)(a+K*1));
        _mm_prefetch(a + 128 + K, _MM_HINT_T0);
        aval[2] = _mm256_loadu_si256((__m256i *)(a+K*2));
        _mm_prefetch(a + 128 + K * 2, _MM_HINT_T0);
        aval[3] = _mm256_loadu_si256((__m256i *)(a+K*3));
        _mm_prefetch(a + 128 + K * 3, _MM_HINT_T0);
        bval =    _mm256_loadu_si256((__m256i *)(b));
        ymm[0]  = _mm256_maddubs_epi16(bval, aval[0]);
        ymm[1]  = _mm256_maddubs_epi16(bval, aval[1]);
        ymm[2]  = _mm256_maddubs_epi16(bval, aval[2]);
        ymm[3]  = _mm256_maddubs_epi16(bval, aval[3]);

        ymm[0]  = _mm256_madd_epi16(ymm[0], one);
        ymm[1]  = _mm256_madd_epi16(ymm[1], one);
        ymm[2]  = _mm256_madd_epi16(ymm[2], one);
        ymm[3]  = _mm256_madd_epi16(ymm[3], one);

        acc[0]  = _mm256_add_epi32(ymm[0], acc[0]);
        acc[1]  = _mm256_add_epi32(ymm[1], acc[1]);
        acc[2]  = _mm256_add_epi32(ymm[2], acc[2]);
        acc[3]  = _mm256_add_epi32(ymm[3], acc[3]);
        a += 32;
        b += 32;
    }
    __m128i xmm[8];
    __m128i offset;
    
    xmm[0] = _mm256_extracti128_si256(acc[0], 0);
    xmm[1] = _mm256_extracti128_si256(acc[0], 1);
    xmm[2] = _mm256_extracti128_si256(acc[1], 0);
    xmm[3] = _mm256_extracti128_si256(acc[1], 1);

    xmm[4] = _mm256_extracti128_si256(acc[2], 0);
    xmm[5] = _mm256_extracti128_si256(acc[2], 1);
    xmm[6] = _mm256_extracti128_si256(acc[3], 0);
    xmm[7] = _mm256_extracti128_si256(acc[3], 1);

    xmm[0] = _mm_add_epi32(xmm[0], xmm[1]);
    xmm[2] = _mm_add_epi32(xmm[2], xmm[3]);
    xmm[4] = _mm_add_epi32(xmm[4], xmm[5]);
    xmm[6] = _mm_add_epi32(xmm[6], xmm[7]);

    xmm[0] = _mm_hadd_epi32(xmm[0], xmm[2]);
    xmm[4] = _mm_hadd_epi32(xmm[4], xmm[6]);
    xmm[0] = _mm_hadd_epi32(xmm[0], xmm[4]);
 
    offset = _mm_loadu_si128((__m128i *)offset_ptr);
    xmm[0] = _mm_add_epi32(xmm[0], offset);
    _mm_storeu_si128((__m128i *)c, xmm[0]);
}


// =================== float =======================

template<int K>
void dot_1x1(const float *a, const float *b, const float *offset_ptr, float *dst){
    __m256 acc;
    __m256 aval, bval;

    acc = _mm256_setzero_ps();

    for (int k = 0; k < K; k+=8) {
        aval = _mm256_loadu_ps(a);
        bval = _mm256_loadu_ps(b);
        acc  =  _mm256_fmadd_ps(aval, bval, acc);

        a += 8;
        b += 8;
    }
    
    __m128 xmm[2];
    
    xmm[0] = _mm256_extractf128_ps(acc, 0);
    xmm[1] = _mm256_extractf128_ps(acc, 1);

    xmm[0] = _mm_add_ps(xmm[0], xmm[1]);

    xmm[0] = _mm_hadd_ps(xmm[0], xmm[0]);
    xmm[0] = _mm_hadd_ps(xmm[0], xmm[0]);

    float f;
    _MM_EXTRACT_FLOAT(f, xmm[0], 0);
    dst[0] = f - offset_ptr[0];
}

template<int K>
void dot_4x1(const float *a, const float *b, const float *offset_ptr, float *dst){
    __m256 acc[4];
    __m256 aval[4], bval;
    for(int i=0; i<4; i++)
        acc[i] = _mm256_setzero_ps();
    for (int k = 0; k < K; k+=8) {
        aval[0] = _mm256_loadu_ps(a);
        aval[1] = _mm256_loadu_ps(a+K*1);
        aval[2] = _mm256_loadu_ps(a+K*2);
        aval[3] = _mm256_loadu_ps(a+K*3);
        bval = _mm256_loadu_ps(b);
        acc[0]  =  _mm256_fmadd_ps(aval[0], bval, acc[0]);
        acc[1]  =  _mm256_fmadd_ps(aval[1], bval, acc[1]);
        acc[2]  =  _mm256_fmadd_ps(aval[2], bval, acc[2]);
        acc[3]  =  _mm256_fmadd_ps(aval[3], bval, acc[3]);

        a += 8;
        b += 8;
    }
    
    __m128 xmm[8];
    __m128 offset;

    offset = _mm_loadu_ps(offset_ptr);
    xmm[0] = _mm256_extractf128_ps(acc[0], 0);
    xmm[1] = _mm256_extractf128_ps(acc[0], 1);
    xmm[2] = _mm256_extractf128_ps(acc[1], 0);
    xmm[3] = _mm256_extractf128_ps(acc[1], 1);

    xmm[4] = _mm256_extractf128_ps(acc[2], 0);
    xmm[5] = _mm256_extractf128_ps(acc[2], 1);
    xmm[6] = _mm256_extractf128_ps(acc[3], 0);
    xmm[7] = _mm256_extractf128_ps(acc[3], 1);

    xmm[0] = _mm_add_ps(xmm[0], xmm[1]);
    xmm[2] = _mm_add_ps(xmm[2], xmm[3]);
    xmm[4] = _mm_add_ps(xmm[4], xmm[5]);
    xmm[6] = _mm_add_ps(xmm[6], xmm[7]);

    xmm[0] = _mm_hadd_ps(xmm[0], xmm[2]);
    xmm[4] = _mm_hadd_ps(xmm[4], xmm[6]);   
    xmm[0] = _mm_hadd_ps(xmm[0], xmm[4]);
    xmm[0] = _mm_sub_ps(xmm[0], offset);
    _mm_storeu_ps(dst    , xmm[0]);
}

template<int K>
void dot_4x2(const float *a, const float *b, const float *offset_ptr, float *dst, const int ldc){
    __m256 acc[8];
    __m256 aval[4], bval[2];
    for(int i=0; i<8; i++)
        acc[i] = _mm256_setzero_ps();
    for (int k = 0; k < K; k+=8) {
        aval[0] = _mm256_loadu_ps(a);
        aval[1] = _mm256_loadu_ps(a+K*1);
        aval[2] = _mm256_loadu_ps(a+K*2);
        aval[3] = _mm256_loadu_ps(a+K*3);
        bval[0] = _mm256_loadu_ps(b);
        bval[1] = _mm256_loadu_ps(b+K);
        acc[0]  =  _mm256_fmadd_ps(aval[0], bval[0], acc[0]);
        acc[1]  =  _mm256_fmadd_ps(aval[1], bval[0], acc[1]);
        acc[2]  =  _mm256_fmadd_ps(aval[2], bval[0], acc[2]);
        acc[3]  =  _mm256_fmadd_ps(aval[3], bval[0], acc[3]);
        acc[4]  =  _mm256_fmadd_ps(aval[0], bval[1], acc[4]);
        acc[5]  =  _mm256_fmadd_ps(aval[1], bval[1], acc[5]);
        acc[6]  =  _mm256_fmadd_ps(aval[2], bval[1], acc[6]);
        acc[7]  =  _mm256_fmadd_ps(aval[3], bval[1], acc[7]);

        a += 8;
        b += 8;
    }
    
    __m128 xmm[16];
    __m128 offset;
    offset = _mm_loadu_ps(offset_ptr);
    xmm[0] = _mm256_extractf128_ps(acc[0], 0);
    xmm[1] = _mm256_extractf128_ps(acc[0], 1);
    xmm[2] = _mm256_extractf128_ps(acc[1], 0);
    xmm[3] = _mm256_extractf128_ps(acc[1], 1);

    xmm[4] = _mm256_extractf128_ps(acc[2], 0);
    xmm[5] = _mm256_extractf128_ps(acc[2], 1);
    xmm[6] = _mm256_extractf128_ps(acc[3], 0);
    xmm[7] = _mm256_extractf128_ps(acc[3], 1);

    xmm[0] = _mm_add_ps(xmm[0], xmm[1]);
    xmm[2] = _mm_add_ps(xmm[2], xmm[3]);
    xmm[4] = _mm_add_ps(xmm[4], xmm[5]);
    xmm[6] = _mm_add_ps(xmm[6], xmm[7]);

    xmm[0] = _mm_hadd_ps(xmm[0], xmm[2]);
    xmm[4] = _mm_hadd_ps(xmm[4], xmm[6]);
    xmm[0] = _mm_hadd_ps(xmm[0], xmm[4]);
    xmm[0] = _mm_sub_ps(xmm[0], offset);
    
    offset = _mm_loadu_ps(offset_ptr + 4);
    xmm[8] = _mm256_extractf128_ps(acc[4], 0);
    xmm[9] = _mm256_extractf128_ps(acc[4], 1);
    xmm[10] = _mm256_extractf128_ps(acc[5], 0);
    xmm[11] = _mm256_extractf128_ps(acc[5], 1);

    xmm[12] = _mm256_extractf128_ps(acc[6], 0);
    xmm[13] = _mm256_extractf128_ps(acc[6], 1);
    xmm[14] = _mm256_extractf128_ps(acc[7], 0);
    xmm[15] = _mm256_extractf128_ps(acc[7], 1);

    xmm[8] = _mm_add_ps(xmm[8], xmm[9]);
    xmm[10] = _mm_add_ps(xmm[10], xmm[11]);
    xmm[12] = _mm_add_ps(xmm[12], xmm[13]);
    xmm[14] = _mm_add_ps(xmm[14], xmm[15]);

    xmm[8] = _mm_hadd_ps(xmm[8], xmm[10]);
    xmm[12] = _mm_hadd_ps(xmm[12], xmm[14]);
    xmm[8] = _mm_hadd_ps(xmm[8], xmm[12]);
    xmm[8] = _mm_sub_ps(xmm[8], offset);
    
    _mm_storeu_ps(dst      , xmm[0]);
    _mm_storeu_ps(dst + ldc, xmm[8]);
}


void dot_nt_1x1(const float *a, const float *b, const float *offset_ptr, const int K, float *dst){
    __m256 acc;
    __m256 aval, bval;

    acc = _mm256_setzero_ps();

    for (int k = 0; k < K; k+=8) {
        aval = _mm256_loadu_ps(a);
        bval = _mm256_loadu_ps(b);
        acc  =  _mm256_fmadd_ps(aval, bval, acc);

        a += 8;
        b += 8;
    }
    
    __m128 xmm[2];
    
    xmm[0] = _mm256_extractf128_ps(acc, 0);
    xmm[1] = _mm256_extractf128_ps(acc, 1);

    xmm[0] = _mm_add_ps(xmm[0], xmm[1]);

    xmm[0] = _mm_hadd_ps(xmm[0], xmm[0]);
    xmm[0] = _mm_hadd_ps(xmm[0], xmm[0]);

    float f;
    _MM_EXTRACT_FLOAT(f, xmm[0], 0);
    dst[0] = f - offset_ptr[0];
}

void dot_nt_4x1(const float *a, const float *b, const float *offset_ptr, const int K, float *dst){
    __m256 acc[4];
    __m256 aval[4], bval;
    for(int i=0; i<4; i++)
        acc[i] = _mm256_setzero_ps();
    for (int k = 0; k < K; k+=8) {
        aval[0] = _mm256_loadu_ps(a);
        aval[1] = _mm256_loadu_ps(a+K*1);
        aval[2] = _mm256_loadu_ps(a+K*2);
        aval[3] = _mm256_loadu_ps(a+K*3);
        bval = _mm256_loadu_ps(b);
        acc[0]  =  _mm256_fmadd_ps(aval[0], bval, acc[0]);
        acc[1]  =  _mm256_fmadd_ps(aval[1], bval, acc[1]);
        acc[2]  =  _mm256_fmadd_ps(aval[2], bval, acc[2]);
        acc[3]  =  _mm256_fmadd_ps(aval[3], bval, acc[3]);

        a += 8;
        b += 8;
    }
    
    __m128 xmm[8];
    __m128 offset;

    offset = _mm_loadu_ps(offset_ptr);
    xmm[0] = _mm256_extractf128_ps(acc[0], 0);
    xmm[1] = _mm256_extractf128_ps(acc[0], 1);
    xmm[2] = _mm256_extractf128_ps(acc[1], 0);
    xmm[3] = _mm256_extractf128_ps(acc[1], 1);

    xmm[4] = _mm256_extractf128_ps(acc[2], 0);
    xmm[5] = _mm256_extractf128_ps(acc[2], 1);
    xmm[6] = _mm256_extractf128_ps(acc[3], 0);
    xmm[7] = _mm256_extractf128_ps(acc[3], 1);

    xmm[0] = _mm_add_ps(xmm[0], xmm[1]);
    xmm[2] = _mm_add_ps(xmm[2], xmm[3]);
    xmm[4] = _mm_add_ps(xmm[4], xmm[5]);
    xmm[6] = _mm_add_ps(xmm[6], xmm[7]);

    xmm[0] = _mm_hadd_ps(xmm[0], xmm[2]);
    xmm[4] = _mm_hadd_ps(xmm[4], xmm[6]);   
    xmm[0] = _mm_hadd_ps(xmm[0], xmm[4]);
    xmm[0] = _mm_sub_ps(xmm[0], offset);
    _mm_storeu_ps(dst    , xmm[0]);
}

void dot_nt_4x2(const float *a, const float *b, const float *offset_ptr, const int K, float *dst, const int ldc){
    __m256 acc[8];
    __m256 aval[4], bval[2];
    for(int i=0; i<8; i++)
        acc[i] = _mm256_setzero_ps();
    for (int k = 0; k < K; k+=8) {
        aval[0] = _mm256_loadu_ps(a);
        aval[1] = _mm256_loadu_ps(a+K*1);
        aval[2] = _mm256_loadu_ps(a+K*2);
        aval[3] = _mm256_loadu_ps(a+K*3);
        bval[0] = _mm256_loadu_ps(b);
        bval[1] = _mm256_loadu_ps(b+K);
        acc[0]  =  _mm256_fmadd_ps(aval[0], bval[0], acc[0]);
        acc[1]  =  _mm256_fmadd_ps(aval[1], bval[0], acc[1]);
        acc[2]  =  _mm256_fmadd_ps(aval[2], bval[0], acc[2]);
        acc[3]  =  _mm256_fmadd_ps(aval[3], bval[0], acc[3]);
        acc[4]  =  _mm256_fmadd_ps(aval[0], bval[1], acc[4]);
        acc[5]  =  _mm256_fmadd_ps(aval[1], bval[1], acc[5]);
        acc[6]  =  _mm256_fmadd_ps(aval[2], bval[1], acc[6]);
        acc[7]  =  _mm256_fmadd_ps(aval[3], bval[1], acc[7]);

        a += 8;
        b += 8;
    }
    
    __m128 xmm[16];
    __m128 offset;
    offset = _mm_loadu_ps(offset_ptr);
    xmm[0] = _mm256_extractf128_ps(acc[0], 0);
    xmm[1] = _mm256_extractf128_ps(acc[0], 1);
    xmm[2] = _mm256_extractf128_ps(acc[1], 0);
    xmm[3] = _mm256_extractf128_ps(acc[1], 1);

    xmm[4] = _mm256_extractf128_ps(acc[2], 0);
    xmm[5] = _mm256_extractf128_ps(acc[2], 1);
    xmm[6] = _mm256_extractf128_ps(acc[3], 0);
    xmm[7] = _mm256_extractf128_ps(acc[3], 1);

    xmm[0] = _mm_add_ps(xmm[0], xmm[1]);
    xmm[2] = _mm_add_ps(xmm[2], xmm[3]);
    xmm[4] = _mm_add_ps(xmm[4], xmm[5]);
    xmm[6] = _mm_add_ps(xmm[6], xmm[7]);

    xmm[0] = _mm_hadd_ps(xmm[0], xmm[2]);
    xmm[4] = _mm_hadd_ps(xmm[4], xmm[6]);
    xmm[0] = _mm_hadd_ps(xmm[0], xmm[4]);
    xmm[0] = _mm_sub_ps(xmm[0], offset);
    
    offset = _mm_loadu_ps(offset_ptr + 4);
    xmm[8] = _mm256_extractf128_ps(acc[4], 0);
    xmm[9] = _mm256_extractf128_ps(acc[4], 1);
    xmm[10] = _mm256_extractf128_ps(acc[5], 0);
    xmm[11] = _mm256_extractf128_ps(acc[5], 1);

    xmm[12] = _mm256_extractf128_ps(acc[6], 0);
    xmm[13] = _mm256_extractf128_ps(acc[6], 1);
    xmm[14] = _mm256_extractf128_ps(acc[7], 0);
    xmm[15] = _mm256_extractf128_ps(acc[7], 1);

    xmm[8] = _mm_add_ps(xmm[8], xmm[9]);
    xmm[10] = _mm_add_ps(xmm[10], xmm[11]);
    xmm[12] = _mm_add_ps(xmm[12], xmm[13]);
    xmm[14] = _mm_add_ps(xmm[14], xmm[15]);

    xmm[8] = _mm_hadd_ps(xmm[8], xmm[10]);
    xmm[12] = _mm_hadd_ps(xmm[12], xmm[14]);
    xmm[8] = _mm_hadd_ps(xmm[8], xmm[12]);
    xmm[8] = _mm_sub_ps(xmm[8], offset);
    
    _mm_storeu_ps(dst      , xmm[0]);
    _mm_storeu_ps(dst + ldc, xmm[8]);
}

template<typename T>
inline void r_dot_prod(const T *A, const T *B, const typemap_t<T> *offset, const int M, const int N, const int K, typemap_t<T> *dst, const int ldc){
    if (K == 512){
        int iA=0, iB=0;
        for (; iB + 3 < N; iB += 4){
            iA = 0;
            for (; iA + 1 < M; iA += 2){
                dot_4x2<512>(B + iB * K, A + iA * K, offset + iB, dst + iA * ldc + iB, ldc);
            }
            for (; iA < M; ++iA) dot_4x1<512>(B + iB * K, A + iA * K, offset + iB, dst + iA * ldc + iB);
        }
        for (; iB < N; ++iB)
        for (iA = 0; iA < M; ++iA) dot_1x1<512>(B + iB * K, A + iA * K, offset + iB, dst + iA * ldc + iB);
    }
    else{
        int iA=0, iB=0;
        for (; iB + 3 < N; iB += 4){
            iA = 0;
            for (; iA + 1 < M; iA += 2){
                dot_nt_4x2(B + iB * K, A + iA * K, offset + iB, K, dst + iA * ldc + iB, ldc);
            }
            for (; iA < M; ++iA) dot_nt_4x1(B + iB * K, A + iA * K, offset + iB, K, dst + iA * ldc + iB);
        }
        for (; iB < N; ++iB)
        for (iA = 0; iA < M; ++iA) dot_nt_1x1(B + iB * K, A + iA * K, offset + iB, K, dst + iA * ldc + iB);
    }
}

}
#endif