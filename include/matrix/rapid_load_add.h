#pragma once
#include "rsearch_def.h"
#include "matrix_load_add.h"
namespace rsearch{
using std::vector;
using std::pair;
template<typename T>
class rapid_load_add : public load_add{
public:    
    rapid_load_add(){}
    virtual ~rapid_load_add(){}
    using Tout = typemap_t<T>;
    virtual int set(int32_t code_len, int32_t topk, int32_t max_batch, int32_t max_block) override;
    virtual int mul(const T* const A, const T* const code_book, int batch, int block, pair<Tout, idx_t> **res) override;

};
}