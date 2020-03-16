#pragma once
#include "rsearch_type.h"
#include "gallery/pqivf_gallery.h"
#include "matrix/rapid_matrix_mul.h"
#include "probe/rsearch_probe.h"
#include "utils/utils.h"

namespace rsearch{
using std::pair;
using std::vector;
using std::make_pair;
template<typename T, DistanceType dist_type> class probe;
template<typename T,
        DistanceType dist_type>
class pqivf_probe : public probe<T, dist_type>{
public:
    pqivf_probe(int dimension, int topk);
    ~pqivf_probe();
    using Tout = typemap_t<T>;
    virtual int create_gallery(gallery<T, dist_type> ** ga_ptr) override;
    virtual int query(const T * const x, const int n, gallery<T, dist_type> * ga, Tout *sims, uint32_t *idx) override;
    virtual int query_with_uids(const T* const x, const int n, gallery<T, dist_type> * ga, uint32_t *uids, const int m, Tout *sims, uint32_t *idx) override;
private:
    matrix_mul<T>* cq_mm, pq_mm;
    int32_t max_batch, max_block, topk, dimension;

    int cq_num;
    int select_cq;
    int pq_dimension;
    int pq_num;
    int code_len;
    int res_cache_size;

    pair<Tout, idx_t>* res;
    uint32_t nprocs;
};
template<typename T,
        DistanceType dist_type>
pqivf_probe<T, dist_type>::pqivf_probe(int dimension, int topk):probe<T, dist_type>(){
    this->nprocs = std::thread::hardware_concurrency();
    this->max_batch = 32;
    this->max_block = 102400;
    this->cq_num = 4096;
    this->select_cq = 2048;
    this->res_cache_size = 1024000;
    this->cq_mm = new rapid_matrix_mul<T>;
    this->pq_mm = new rapid_matrix_mul<T>;
    this->cq_mm->set(this->dimension, this->select_cq, max_batch, max_block);
    this->pq_mm->set(this->pq_dimension, topk, max_batch * this->code_len, this->pq_num);
    this->code_len = this->dimension / this->pq_dimension;
    this->res = (pair<Tout, idx_t>*)malloc(this->max_batch * this->max_block * sizeof(pair<Tout, idx_t>));
}

template<typename T,
        DistanceType dist_type>
pqivf_probe<T, dist_type>::~pqivf_probe(){
    delete this->cq_mm;
    delete this->pq_mm;
}
template<typename T,
        DistanceType dist_type>
int pqivf_probe<T, dist_type>::create_gallery(gallery<T, dist_type> ** ga_ptr){
    struct pqivf_traits traits(4096,2048,32,256);
    pqivf_gallery<T, dist_type> * ga = new pqivf_gallery<T, dist_type>(this->dimension, traits);
    (*ga_ptr) = (gallery<T, dist_type>*)ga;
    return 0;
}

template<typename T>
inline void get_value(int8_t* data, pair<T, idx_t>* code_book, int code_len, int qid, int block, pair<T, idx_t>* res){
    for (int i = 0; i < block; ++i){
        int v = 0;
        for (int j = 0; j < code_len; ++j)
            v += code_book[j];
        res[i].first = v;
        res[i].second = qid + i;
    }
}


template<typename T,
        DistanceType dist_type>
int pqivf_probe<T, dist_type>::query(const T * const x, const int n, gallery<T, dist_type> * ga, Tout *sims, uint32_t *idx){

    pqivf_gallery<T, dist_type>* c_ga = (pqivf_gallery<T, dist_type>*)ga;
    vector<vector<pair<Tout, idx_t> > > ans(this->max_batch);
    if (c_ga->num < this->topk){
        ans.reserve(n);
        for (int i = 0; i < n; ++i){
            for (int j = 0; j < this->cq_num; ++j)
                for (int k = 0; k < c_ga->data[j].size(); ++k)
                    ans[i].push_back(std::make_pair(vec_dis<T, dist_type>(x + 1LL * i * this->dimension, c_ga->data[j].data() + 1LL * k * this->dimension, this->dimension), k));
            sort(ans[i].begin(), ans[i].end(), pair_greator<T, idx_t>());
            for (int j = 0; j < c_ga->num; ++j){
                sims[i * this->topk + j] = ans[i][j].first;
                idx[i * this->topk + j] = c_ga->ids[ans[i][j].second];
            }
            for (int j = c_ga->num; j < this->topk; ++j){
                sims[i * this->topk + j] = 0;
                idx[i * this->topk + j] = -1;
            }
        }
        return 0;
    }

    pair<Tout, idx_t>* cq_res;
    pair<Tout, idx_t>* pq_res;
    for (int i = 0; i < n; i += this->max_batch){
        this->cq_mm.mul(x, this->cq.data(), this->cq_offset.data(), this->max_batch, this->cq_num, &cq_res);

        //get code_book
        this->pq_mm.mul(x, this->pq.data(), this->pq_offset.data(), this->max_batch * this->code_len, this->pq_num, &pq_res);

        for (int j = 0; j < std::min(this->max_batch, n-i); ++j){
            for (int cq_id = 0; cq_id < this->select_cq; ++cq_id){
                uint8_t* data= c_ga->data[cq_res[cq_id].second].data();
                int num = c_ga->block_num[cq_res[cq_id].second];
                for (int vec_id = 0; vec_id < num; vec_id += this->max_block){
                    get_value<Tout>(data + vec_id * this->dimension, pq_res, this->code_len, i + j,
                     std::min(num - vec_id, this->max_block), &this->res[(i + j) * this->max_block + vec_id]);

                    nth_element(res[(i + j) * this->max_block], res[(i + j + 1) * this->max_block], pair_greator<Tout, int>());
                    for (int k = 0; k < this->topk; ++k)
                        ans[j].push_back(res[k]);
                }
            }
            nth_element(ans[j].data(), ans[j].data() + this->topk + 1, pair_greator<Tout, int>());
            for (int k =0 ; k < this->topk; ++k){
                sims[j * this->topk + k] = ans[j][k].first;
                idx[j * this->topk + k] = ans[j][k].second;
            }
        }
    }
    return 0;
}

template<typename T,
        DistanceType dist_type>
int pqivf_probe<T, dist_type>::query_with_uids(const T* const x, const int n, gallery<T, dist_type> * ga, uint32_t *uids, const int m, Tout *sims, uint32_t *idx){
    /* TODO */
    return 0;
}

}
