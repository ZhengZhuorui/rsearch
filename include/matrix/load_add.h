#pragma once
#include "rsearch_def.h"
namespace rsearch{
using std::vector;
using std::pair;
template<typename T>
class load_add{
public:    
    load_add(){}
    virtual ~load_add(){}
    using Tout = typemap_t<T>;
    virtual int set(int32_t code_len, int32_t topk, int32_t max_batch, int32_t max_block) = 0;
    virtual int mul(const T* const A, const T* const code_book, int batch, int block, pair<Tout, idx_t> **res) = 0;

};
}