#include "matrix/rapid_matrix_mul.h"
namespace rsearch{
template<typename T>
int rapid_matrix_mul<T>::set(int32_t dimension, int32_t topk, int32_t max_batch, int32_t max_block){
    this->mtx.lock();
    this->max_batch = max_batch;
    this->max_block = max_block;
    this->dimension = dimension;
    this->topk = topk;
    {
    if (this->value != NULL)
        free(this->value);
    if (this->topk_value != NULL)
        free(this->topk_value);
    if (this->topk_index != NULL)
        free(this->topk_index);
    if (this->res != NULL)
        free(this->res);
    }
    this->value = (Tout*)malloc(1LL * max_batch * max_block * sizeof(Tout));
    this->topk_value = (Tout*)malloc(1LL * max_batch * topk * sizeof(Tout));
    this->topk_index = (idx_t*)malloc(1LL * max_batch * topk * sizeof(idx_t));
    this->res = (pair<Tout, idx_t>*)malloc(1LL * max_batch * topk * sizeof(pair<Tout, idx_t>));
    this->mtx.unlock();
    return 0;
}
template int rapid_matrix_mul<int8_t>::set(int32_t dimension, int32_t topk, int32_t max_batch, int32_t max_block);
template int rapid_matrix_mul<float>::set(int32_t dimension, int32_t topk, int32_t max_batch, int32_t max_block);

template<typename T>
int rapid_matrix_mul<T>::mul(const T* const A, const T* const B, const Tout* const offset, int batch, int block, pair<Tout, idx_t> **res){
    //if (batch > this->max_batch || block > this->max_block)
    //    return SIZE_TOO_BIG;
    this->mtx.lock();
    {
        r_dot_prod<T>(A, B, offset, batch, block, this->dimension, this->value, this->max_block);
    }
    {
        cpu_select_kv(this->value, this->topk_value, this->topk_index, this->topk, block, batch, this->max_block, true);
    }
    for (int i = 0 ; i < batch; ++i){
        for (int j = 0 ; j < topk; ++j){
            this->res[i * this->topk + j].first = this->topk_value[i * this->topk + j];
            this->res[i * this->topk + j].second = this->topk_index[i * this->topk + j];
        }
    }
    (*res) = this->res;
    this->mtx.unlock();
    return 0;
}
template int rapid_matrix_mul<int8_t>::mul(const int8_t* const A, const int8_t* const B, const int* const offset, int batch, int block, pair<int, idx_t> **res);
template int rapid_matrix_mul<float>::mul(const float* const A, const float* const B, const float* const offset, int batch, int block, pair<float, idx_t> **res);
}
