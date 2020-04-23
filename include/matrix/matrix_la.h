#pragma once
#include "rsearch_def.h"
namespace rsearch{
using std::vector;
using std::pair;
template<typename T>
class matrix_la{
public:    
    matrix_la(){}
    virtual ~matrix_la(){}
    virtual int set(int32_t code_len, int32_t topk, int32_t max_batch, int32_t max_block, int32_t codebook_size) = 0;
    virtual int la(const int32_t* const A, const T* const code_book, int batch, int block, pair<T, idx_t> **res)  = 0;

};
}