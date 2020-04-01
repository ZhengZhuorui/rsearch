#pragma once
#include "rsearch_def.h"
#include "matrix/matrix_la.h"
namespace rsearch{
using std::vector;
using std::pair;
template<typename T>
class rapid_matrix_la : public matrix_la<T>{
public:    
    rapid_matrix_la(){
        this->value = NULL;
        this->topk_value = NULL;
        this->topk_index = NULL;
        this->res = NULL;
    }
    virtual ~rapid_matrix_la(){
        if (this->value != NULL)
            free(this->value);
        if (this->topk_value != NULL)
            free(this->topk_value);
        if (this->topk_index != NULL)
            free(this->topk_index);
        if (this->res != NULL)
            free(this->res);
    }
    virtual int set(int32_t code_len, int32_t topk, int32_t max_batch, int32_t max_block, int32_t code_per_dimension) override;
    virtual int la(const int32_t* const A, const T* const code_book, int batch, int block, pair<T, idx_t> **res) override;

private:
    T* value;
    T* topk_value;
    idx_t* topk_index;
    pair<T, idx_t>* res;
    int32_t max_batch, max_block, dimension, topk;
    int32_t code_book_size;
    //std::mutex mtx;
};
}