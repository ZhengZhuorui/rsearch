#pragma once
#include "rsearch_def.h"
namespace rsearch{
using std::vector;
using std::pair;
template<typename T>
class matrix_mul{
public:    
    matrix_mul(){}
    virtual ~matrix_mul(){}
    using Tout = typemap_t<T>;
    virtual int set(int32_t dimension, int32_t topk, int32_t max_batch, int32_t max_block) = 0;
    virtual int mul(const T* const A, const T* const B, const Tout* const offset, int batch, int block, pair<Tout, idx_t> **res) = 0;

};
}