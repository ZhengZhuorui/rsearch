#include "matrix/base_matrix_mul.h"
namespace rsearch{
template<typename T>
int base_matrix_mul<T>::set(int32_t dimension, int32_t topk, int32_t max_batch, int32_t max_block){
    this->max_batch = max_batch;
    this->max_block = max_block;
    this->dimension = dimension;
    this->topk = topk;
    this->value = (Tout*)malloc(max_block * max_batch * sizeof(Tout));
    this->topk_index = (idx_t*)malloc(max_batch * topk * sizeof(idx_t));
    this->topk_value = (Tout*)malloc(max_batch * topk * sizeof(Tout));
    this->res = (pair<Tout, idx_t>*)malloc(max_block * topk * sizeof(pair<Tout, idx_t>));
    return 0;
}
template int base_matrix_mul<int8_t>::set(int32_t dimension, int32_t topk, int32_t max_batch, int32_t max_block);
template int base_matrix_mul<float>::set(int32_t dimension, int32_t topk, int32_t max_batch, int32_t max_block);

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
    {
        cpu_select_kv(this->value, this->topk_value, this->topk_index, this->topk, block, batch, true);
    }
    for (int i = 0 ; i < batch; ++i){
        for (int j = 0 ; j < topk; ++j){
            this->res[i * this->topk + j].first = this->topk_value[i * this->topk + j];
            this->res[i * this->topk + j].second = this->topk_index[i * this->topk + j];
        }
    }
    (*res) = this->res;
    return 0;
}
template int base_matrix_mul<int8_t>::mul(const int8_t* const A, const int8_t* const B, const int* const offset, int batch, int block, pair<int, idx_t> **res);
template int base_matrix_mul<float>::mul(const float* const A, const float* const B, const float* const offset, int batch, int block, pair<float, idx_t> **res);

}