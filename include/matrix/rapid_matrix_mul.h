#pragma once
#include "matrix/matrix_mul.h"
#include "rsearch_type.h"
#include <bits/stdc++.h>
#include "utils/utils.h"
#ifdef  __SSE__
#include "utils/avx2_asm.h"

#include <typeinfo>
#include <stdio.h>
#include <stdint.h>
#include <mmintrin.h>
#include <xmmintrin.h>
#include <pmmintrin.h>
#include <emmintrin.h>
#include <smmintrin.h>
#include <immintrin.h>
namespace rsearch{

using std::vector;
using std::pair;
template<typename T>
class rapid_matrix_mul : public matrix_mul<T>{
public:
    rapid_matrix_mul():matrix_mul<T>(){}
    virtual ~rapid_matrix_mul(){
        free(this->value);
        free(this->res);
    }
    using Tout = typemap_t<T>;
    virtual int set(int32_t dimension, int32_t topk, int32_t max_batch, int32_t max_block);
    virtual int mul(const T* const A, const T* const B, const Tout* const offset, int batch, int block, pair<Tout, idx_t> **res);
private:
    Tout* value;
    pair<Tout, idx_t>* res;
    int32_t max_batch, max_block, dimension, topk;
};

template<typename T>
int rapid_matrix_mul<T>::set(int32_t dimension, int32_t topk, int32_t max_batch, int32_t max_block){
    this->max_batch = max_batch;
    this->max_block = max_block;
    this->dimension = dimension;
    this->topk = topk;
    this->value = (Tout*)malloc(max_block * max_batch * sizeof(Tout));
    this->res = (pair<Tout, idx_t>*)malloc(max_block * max_batch * sizeof(pair<Tout, idx_t>));
    return 0;
}
template<typename T>
int rapid_matrix_mul<T>::mul(const T* const A, const T* const B, const Tout* const offset, int batch, int block, pair<Tout, idx_t> **res){
    int iA = 0, iB = 0;
    if (is_same_type<T, float>() == true){
        if (this->dimension == 512)
        {
            for (; iB + 3 < block; iB += 4){
                iA = 0;
                for (; iA + 1 < batch; iA += 2){
                    dot_4x2<512>(B + iB * this->dimension, A + iA * this->dimension, offset + iB, this->value + iA * this->max_block + iB, this->max_block);
                }
                for (; iA < batch; ++iA) dot_4x1<512>(B + iB * this->dimension, A + iA * this->dimension, offset + iB, this->value + iA * this->max_block + iB);
            }
            for (; iB < block; ++iB)
            for (iA = 0; iA < batch; ++iA) dot_1x1<512>(B + iB * this->dimension, A + iA * this->dimension, offset + iB, this->value + iA * this->max_block + iB);
        }
    }

    for (int i = 0 ; i < batch; ++i){
        for (int j = 0 ; j < block; ++j){
            this->res[i * this->max_block + j] = std::make_pair(this->value[i * this->max_block + j], j);
        }
        std::nth_element(this->res + i * this->max_block, this->res + i * this->max_block + topk + 1, this->res + (i + 1) * this->max_block, pair_greator<Tout, idx_t>());
    }
    (*res) = this->res;
    return 0;
}
}

#endif
