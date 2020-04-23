#pragma once
#include "rsearch_def.h"
#include "probe/cpu_base_probe.h"
#include "probe/pqivf_probe.h"
#include "probe/cpu_base_mt_probe.h"
#include "probe/faiss_probe.h"

namespace rsearch{

template<typename T>
class probe{
public:
    probe(){}
    virtual ~probe(){}
    using Tout = typemap_t<T>;
    virtual int create_gallery(gallery<T> ** ga_ptr) = 0;
    virtual int query(const T * const x, const int n, gallery<T> * ga, Tout *sims, idx_t *idx) = 0;
    virtual int query_with_uids(const T* const x, const int n, gallery<T> * ga, idx_t *uids, const int m, Tout *sims, idx_t *idx) = 0;
};

template<typename T,
        DistanceType dist_type,
        typename matrix_type>
class base_probe : public probe<T>{
public:
    base_probe() : probe<T>(){}
    virtual ~base_probe(){}
    using Tout = typemap_t<T>;
    virtual int create_gallery(gallery<T> ** ga_ptr) = 0;
    virtual int query(const T * const x, const int n, gallery<T> * ga, Tout *sims, idx_t *idx) = 0;
    virtual int query_with_uids(const T* const x, const int n, gallery<T> * ga, idx_t *uids, const int m, Tout *sims, idx_t *idx) = 0;
};

template<typename T>
probe<T>* create_probe(int dimension, int topk, DistanceType dist_type, MethodType method_type);

}

