#pragma once
#include "rsearch_def.h"
#include "probe/rsearch_probe.h"
#include "matrix/base_matrix_mul.h"
#include "matrix/rapid_matrix_mul.h"
#include "gallery/cpu_base_gallery.h"
#include "utils/utils.h"
#include "utils/ThreadPool.h"
namespace rsearch{

template<typename T,
        DistanceType dist_type,
        typename matrix_type>
class cpu_base_mt_probe : public base_probe<T, dist_type, matrix_type>{
public:
    cpu_base_mt_probe(int dimension, int topk);
    ~cpu_base_mt_probe();
    using Tout = typemap_t<T>;
    virtual int create_gallery(gallery<T, dist_type> ** ga_ptr) override;
    virtual int query(const T * const x, const int n, gallery<T, dist_type> * ga, Tout *sims, idx_t *idx) override;
    virtual int query_with_uids(const T* const x, const int n, gallery<T, dist_type> * ga, idx_t *uids, const int m, Tout *sims, idx_t *idx) override;
private:
    virtual void query_bunch(const int mm_id, const T* x, const T* data, const Tout* offset, const int batch, const int block, const int base_id);
    std::vector<matrix_mul<T>*> mm;
    int32_t max_batch, max_block, topk, dimension;
    std::vector<T> x_tmp;
    ThreadPool* threadpool;
    int nprocs;
    std::mutex mtx;
    vector<vector<pair<Tout, idx_t> > > ans;
};

}