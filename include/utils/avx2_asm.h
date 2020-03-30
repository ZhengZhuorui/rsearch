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
#include <avx2intrin.h>

#include "rsearch_type.h"
namespace rsearch{
// ============== int8 =============== 

/*
*   for i = 0 to 3 do
*       for j = 0 to 1 do
*           c[j * ldc + i] = a[i] * b[j] + offset_ptr[i]
*/
template<int K>
void dot_4x2(const int8_t *a, const int8_t *b, const int * offset_ptr, int *c, int ldc);

template<int K>
void dot_1x1(const int8_t *a, const int8_t *b, const int * offset_ptr, int *c);

template<int K>
void dot_4x1(const int8_t *a, const int8_t *b, const int * offset_ptr, int *c);

void dot_nt_4x2(const int8_t *a, const int8_t *b, const int * offset_ptr, const int K, int *c, int ldc);

void dot_nt_1x1(const int8_t *a, const int8_t *b, const int * offset_ptr, const int K, int *c);

void dot_nt_4x1(const int8_t *a, const int8_t *b, const int * offset_ptr, const int K, int *c);

// =================== float =======================

template<int K>
void dot_1x1(const float *a, const float *b, const float *offset_ptr, float *dst);

template<int K>
void dot_4x1(const float *a, const float *b, const float *offset_ptr, float *dst);

template<int K>
void dot_4x2(const float *a, const float *b, const float *offset_ptr, float *dst, const int ldc);


void dot_nt_1x1(const float *a, const float *b, const float *offset_ptr, const int K, float *dst);

void dot_nt_4x1(const float *a, const float *b, const float *offset_ptr, const int K, float *dst);

void dot_nt_4x2(const float *a, const float *b, const float *offset_ptr, const int K, float *dst, const int ldc);

template<typename T>
void r_dot_prod(const T *A, const T *B, const typemap_t<T> *offset, const int M, const int N, const int K, typemap_t<T> *dst, const int ldc);

template<int K>
void ld_add_4x1(const int *mem, const int32_t* index, const int* dst);

template<int K>
void ld_add_1x1(const int *mem, const int32_t* index, const int* dst);

template<int K>
void ld_add_4x1(const float *mem, const int32_t* index, const float* dst);

template<int K>
void ld_add_1x1(const float *mem, const int32_t* index, const float* dst);

template<typename T>
void r_ld_add(const T *mem, const int32_t* index, T* dst, const int M,const int N, const int K, const int ldc);

}
#endif