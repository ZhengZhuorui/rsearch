#pragma once
#include "rsearch_def.h"
#include "probe/rsearch_probe.h"
#include "matrix/rapid_matrix_mul.h"
#include "gallery/pqivf_gallery.h"
#include "utils/utils.h"
#include "matrix/rapid_matrix_la.h"
#include "utils/ThreadPool.h"
#include "utils/MthManager.h"
namespace rsearch{
using std::pair;
using std::vector;
using std::make_pair;
template<typename T,
        DistanceType dist_type>
class pqivf_mt_probe : public probe<T>{
public:
    pqivf_mt_probe(int dimension, int topk);
    ~pqivf_mt_probe();
    using Tout = typemap_t<T>;
    virtual int create_gallery(gallery<T> ** ga_ptr) override;
    virtual int query(const T * const x, const int n, gallery<T> * ga, Tout *sims, idx_t *idx) override;
    virtual int query_with_uids(const T* const x, const int n, gallery<T> * ga, idx_t *uids, const int m, Tout *sims, idx_t *idx) override;
private:
    virtual void query_bunch(const int thread_id, const int* data, const Tout* code_book, const int block, const int id, const int offset);
    matrix_mul<T>* cq_mm;
    std::vector<matrix_la<Tout>*> mtx_la;
    int max_batch, max_block, topk, dimension;

    int nprocs;
    int cq_num;
    int select_cq;
    int pq_dimension;
    int pq_num;
    int code_len;
    int res_cache_size;
    int codebook_size;

    Tout* code_book;
    int* prefix;
    
    pair<Tout,idx_t>* merge_cache;

    vector<T> x_tmp;
    vector<T> x_tmp_div;
    vector<Tout> x_offset;
    vector<std::pair<Tout, idx_t> > ans;

    ThreadPool* threadpool;
    MthManager* mth_manager;

};

}
