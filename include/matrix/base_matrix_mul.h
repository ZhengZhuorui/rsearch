#pragma once
#include "matrix/matrix_mul.h"
#include "rsearch_type.h"
#include <bits/stdc++.h>
#include "utils/utils.h"
namespace rsearch{
using std::vector;
using std::pair;
template<typename T>
class base_matrix_mul : public matrix_mul<T>{
public:
    base_matrix_mul():matrix_mul<T>(){}
    virtual ~base_matrix_mul(){
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
int base_matrix_mul<T>::set(int32_t dimension, int32_t topk, int32_t max_batch, int32_t max_block){
    this->max_batch = max_batch;
    this->max_block = max_block;
    this->dimension = dimension;
    this->topk = topk;
    this->value = (T*)malloc(max_block * max_batch * sizeof(T));
    this->res = (pair<Tout, idx_t>*)malloc(max_block * max_batch * sizeof(pair<Tout, idx_t>));
    return 0;
}

template<typename T>
int base_matrix_mul<T>::mul(const T* const A, const T* const B, const Tout* const offset, int batch, int block, pair<Tout, idx_t> **res){
    for (int i = 0 ; i < batch; ++i){
        for (int j = 0 ; j < block; ++j){
            Tout v = 0;
            for (int k = 0 ; k < this->dimension; ++k)
                v += A[i * this->dimension + k] * B[j * this->dimension + k];
            this->value[i * this->max_block + j]= v + offset[j];
        }
    }
    for (int i = 0 ; i < batch; ++i){        
        for (int j = 0 ; j < block; ++j) {
            this->res[i * this->max_block + j] = std::make_pair(this->value[i * this->max_block + j], j);
        }
        std::nth_element(this->res + i * this->max_block, this->res + i * this->max_block + topk + 1, this->res + (i + 1) * this->max_block, pair_greator<Tout, idx_t>());
    }
    (*res) = this->res;
    return 0;
}

}
