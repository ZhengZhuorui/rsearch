#pragma once
#include "rsearch_type.h"
#include "utils/utils.h"
namespace rsearch{
template<typename T,
        DistanceType dist_type>
class gallery{
public:
    gallery(){}

    virtual ~gallery(){}

    virtual int init() = 0;

    using Tout = typemap_t<T>;

    virtual int add(const T* const x, const int n) = 0;

    virtual int add_with_uids(const T* const x, const idx_t * const uids, const int n) = 0;

    virtual int change_by_uids(const T* const x, const idx_t * const uids, const int n) = 0;

    virtual int remove_by_uids(const idx_t * const uids, const int n) = 0;

    virtual int query_by_uids(const idx_t* const uid, const int n, T * x) = 0;

    virtual int reset() = 0;
    
};

}
