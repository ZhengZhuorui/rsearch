#pragma once
#include "rsearch_type.h"
#include "gallery/rsearch_gallery.h"
#include "probe/cpu_base_probe.h"
#include "probe/pqivf_probe.h"

namespace rsearch{

template<typename T,
        DistanceType dist_type>
class probe{
public:
    probe(){}
    virtual ~probe(){}
    using Tout = typemap_t<T>;
    virtual int create_gallery(gallery<T, dist_type> ** ga_ptr) = 0;
    virtual int query(const T * const x, const int n, gallery<T, dist_type> * ga, Tout *sims, uint32_t *idx) = 0;
    virtual int query_with_uids(const T* const x, const int n, gallery<T, dist_type> * ga, uint32_t *uids, const int m, Tout *sims, uint32_t *idx);
};

template<typename T,
        DistanceType dist_type,
        typename matrix_type>
class base_probe : probe<T, dist_type>{
public:
    base_probe() : probe<T, dist_type>(){}
    virtual ~base_probe(){}
    using Tout = typemap_t<T>;
    virtual int create_gallery(gallery<T, dist_type> ** ga_ptr) = 0;
    virtual int query(const T * const x, const int n, gallery<T, dist_type> * ga, Tout *sims, uint32_t *idx) = 0;
    virtual int query_with_uids(const T* const x, const int n, gallery<T, dist_type> * ga, uint32_t *uids, const int m, Tout *sims, uint32_t *idx);
};

template<typename T, 
        DistanceType dist_type>
probe<T, dist_type>* create_probe(int dimension, int topk, MethodType method_type){
    probe<T> * r;
    switch(method_type){
        case X86_DUMMY:
            r = new cpu_base_probe<T, dist_type, base_matrix_mul>(dimension, topk);
            break;
        case X86_RAPID:
            r = new cpu_base_probe<T, dist_type, rapid_matrix_mul>(dimension, topk);
            break;
        case PQIVF:
            //r = new cpu_base_probe<T, dist_type, rapid_matrix_mul, 32, 102400>(dimension, topk);
            break;
    }

    return r;
}

}

