#include "matrix/rapid_matrix_la.h"
#include "utils/avx2_asm.h"
#include "utils/topk_select.h"
namespace rsearch{
template<typename T>
int rapid_matrix_la<T>::set(int32_t dimension, int32_t topk, int32_t max_batch, int32_t max_block, int32_t codebook_size){
    //this->mtx.lock();
    this->max_batch = max_batch;
    this->max_block = max_block;
    this->dimension = dimension;
    this->topk = topk;
    this->codebook_size = codebook_size;
    {
    if (this->value != NULL){
        free(this->value);
        this->value = NULL;
    }
    if (this->topk_value != NULL){
        free(this->topk_value);
        this->topk_value = NULL;
    }
    if (this->topk_index != NULL){
        free(this->topk_index);
        this->topk_index = NULL;
    }
    if (this->res != NULL){
        free(this->res);
        this->res = NULL;
    }
    }
    this->value = (T*)malloc(1LL * max_batch * max_block * sizeof(T));
    this->topk_value = (T*)malloc(1LL * max_batch * topk * sizeof(T));
    this->topk_index = (idx_t*)malloc(1LL * max_batch * topk * sizeof(idx_t));
    this->res = (pair<T, idx_t>*)malloc(1LL * max_batch * topk * sizeof(pair<T, idx_t>));
    //this->mtx.unlock();
    return 0;
}
template int rapid_matrix_la<int>::set(int32_t dimension, int32_t topk, int32_t max_batch, int32_t max_block, int32_t codebook_size);
template int rapid_matrix_la<float>::set(int32_t dimension, int32_t topk, int32_t max_batch, int32_t max_block, int32_t codebook_size);

template<typename T>
int rapid_matrix_la<T>::la(const int32_t* const A, const T* const code_book, int batch, int block, pair<T, idx_t> **res){
    //if (batch > this->max_batch || block > this->max_block)
    //    return SIZE_TOO_BIG;
    //this->mtx.lock();
    {
        r_ld_add(code_book, A, this->value, batch, block, this->dimension, this->codebook_size, this->max_block);
    }
    {
        cpu_select_kv(this->value, this->topk_value, this->topk_index, this->topk, block, batch, this->max_block, true);
    }
    /*static int m = 0;
    if (m < 1024){
        std::cout << this->topk_value[0] << std::endl;
        ++m;
    }*/
    if (block == 660)
        std::cout << "target 2" << std::endl;
    for (int i = 0 ; i < batch; ++i){
        for (int j = 0 ; j < topk; ++j){
            this->res[i * this->topk + j].first = this->topk_value[i * this->topk + j];
            this->res[i * this->topk + j].second = this->topk_index[i * this->topk + j];
        }
    }
    (*res) = this->res;
    //this->mtx.unlock();
    return 0;
}
template int rapid_matrix_la<int>::la(const int32_t* const A, const int* const code_book, int batch, int block, pair<int, idx_t> **res);
template int rapid_matrix_la<float>::la(const int32_t* const A, const float* const code_book, int batch, int block, pair<float, idx_t> **res);
}
