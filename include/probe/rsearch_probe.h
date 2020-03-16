#pragma once
#include "rsearch_type.h"
#include "gallery/rsearch_gallery.h"
#include "probe/cpu_base_probe.h"
#include "probe/pqivf_probe.h"
#include "matrix/base_matrix_mul.h"
#include "matrix/rapid_matrix_mul.h"

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
class base_probe : public probe<T, dist_type>{
public:
    base_probe() : probe<T, dist_type>(){}
    virtual ~base_probe(){}
    using Tout = typemap_t<T>;
    virtual int create_gallery(gallery<T, dist_type> ** ga_ptr) = 0;
    virtual int query(const T * const x, const int n, gallery<T, dist_type> * ga, Tout *sims, uint32_t *idx) = 0;
    virtual int query_with_uids(const T* const x, const int n, gallery<T, dist_type> * ga, uint32_t *uids, const int m, Tout *sims, uint32_t *idx);
};

template<typename T, DistanceType dist_type> class pqivf_probe;
template<typename T, DistanceType dist_type, typename matrix_type> class cpu_base_probe;

template<typename T, 
        DistanceType dist_type>
probe<T, dist_type>* create_probe(int dimension, int topk, MethodType method_type){
    probe<T, dist_type> * r;
    switch(method_type){
        case DUMMY:
            r = new cpu_base_probe<T, dist_type, base_matrix_mul<T> >(dimension, topk);
            break;
        case X86_RAPID:
            r = new cpu_base_probe<T, dist_type, rapid_matrix_mul<T> >(dimension, topk);
            break;
        case X86_PQIVF:
            r = new pqivf_probe<T, dist_type>(dimension, topk);
            break;
    }

    return r;
}

}

