#include "matrix/rapid_matrix_la.h"
#include "utils/avx2_asm.h"
#include "utils/topk_select.h"
namespace rsearch{
template<typename T>
int rapid_matrix_la<T>::set(int32_t dimension, int32_t topk, int32_t max_batch, int32_t max_block, int32_t code_per_dimension){
    this->max_batch = max_batch;
    this->max_block = max_block;
    this->dimension = dimension;
    this->topk = topk;
    this->code_book_size = code_per_dimension * this->dimension;
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
    this->value = (T*)malloc(1LL * max_batch * max_block * sizeof(T));
    this->topk_value = (T*)malloc(1LL * max_batch * topk * sizeof(T));
    this->topk_index = (idx_t*)malloc(1LL * max_batch * topk * sizeof(idx_t));
    this->res = (pair<T, idx_t>*)malloc(1LL * max_batch * topk * sizeof(pair<T, idx_t>));

    return 0;
}
template int rapid_matrix_la<int8_t>::set(int32_t dimension, int32_t topk, int32_t max_batch, int32_t max_block, int32_t code_per_dimension);
template int rapid_matrix_la<float>::set(int32_t dimension, int32_t topk, int32_t max_batch, int32_t max_block, int32_t code_per_dimension);

template<typename T>
int rapid_matrix_la<T>::la(const T* const A, const T* const code_book, int batch, int block, pair<T, idx_t> **res){
    //if (batch > this->max_batch || block > this->max_block)
    //    return SIZE_TOO_BIG;
    {
        r_ld_add(code_book, A, this->value, batch, block, this->dimension, this->code_book_size);
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
    return 0;
}
template int rapid_matrix_la<int>::la(const int* const A, const int* const code_book, int batch, int block, pair<int, idx_t> **res);
template int rapid_matrix_la<float>::la(const float* const A, const float* const code_book, int batch, int block, pair<float, idx_t> **res);
}
