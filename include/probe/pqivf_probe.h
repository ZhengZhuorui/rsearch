#pragma once
#include "rsearch_def.h"
#include "utils/utils.h"

namespace rsearch{
using std::pair;
using std::vector;
using std::make_pair;
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
    matrix_mul<T>* cq_mm;
    //matrix_mul<T>* pq_mm;
    int32_t max_batch, max_block, topk, dimension;

    int cq_num;
    int select_cq;
    int pq_dimension;
    int pq_num;
    int code_len;
    int res_cache_size;

    pair<Tout, idx_t>* res;
    Tout* code_book;
    int32_t* prefix;
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
    this->res_cache_size = 10240;
    this->cq_mm = new rapid_matrix_mul<T>;
    //this->pq_mm = new rapid_matrix_mul<T>;
    this->cq_mm->set(this->dimension, this->select_cq, max_batch, max_block);
    //this->pq_mm->set(this->pq_dimension, topk, max_batch * this->code_len, this->pq_num);
    this->code_len = this->dimension / this->pq_dimension;
    this->res = (pair<Tout, idx_t>*)malloc(this->res_cache_size * sizeof(pair<Tout, idx_t>));
    memset(this->res, 0, this->res_cache_size * sizeof(pair<Tout, idx_t>));
    this->code_book = (Tout*)malloc(this->max_batch * this->pq_num * this->code_len * sizeof(Tout));
    this->prefix = (int32_t*)malloc((this->cq_num + 5) * sizeof(int32_t));
}

template<typename T,
        DistanceType dist_type>
pqivf_probe<T, dist_type>::~pqivf_probe(){
    delete this->cq_mm;
    //delete this->pq_mm;
    free(this->res);
    free(this->prefix);
    free(this->code_book);
}
template<typename T,
        DistanceType dist_type>
int pqivf_probe<T, dist_type>::create_gallery(gallery<T, dist_type> ** ga_ptr){
    struct pqivf_traits traits = {4096,2048,32,256};
    pqivf_gallery<T, dist_type> * ga = new pqivf_gallery<T, dist_type>(this->dimension, traits);
    (*ga_ptr) = (gallery<T, dist_type>*)ga;
    return 0;
}

template<typename T>
inline void get_res(uint8_t* data, T* code_book, int code_len, int ldc, int st, int qid, int block, pair<T, idx_t>* res){
    for (int i = st, _i = 0; i < st + block; ++i, _i += code_len)
        for (int j = 0, _j = 0; j < code_len; ++j, _j += ldc)
            res[i].first += code_book[_j + data[_i + j]];

    for (int i = st, _i = qid; i < st + block; ++i, ++_i) 
        res[i].second = _i;
}

template<typename T,
        DistanceType dist_type>
int pqivf_probe<T, dist_type>::query(const T * const x, const int n, gallery<T, dist_type> * ga, Tout *sims, uint32_t *idx){

    pqivf_gallery<T, dist_type>* c_ga = (pqivf_gallery<T, dist_type>*)ga;
    vector<pair<Tout, idx_t> > ans;
    if (c_ga->num < this->topk){
        for (int i = 0; i < n; ++i)
        for (int j = 0; j < this->topk; ++j){
            sims[i * this->topk + j] = 0;
            idx[i * this->topk + j] = -1;
        }
        return NO_SUPPORT_NUM_LT_K;
    }

    pair<Tout, idx_t>* cq_res;
    //pair<Tout, idx_t>* pq_res;
    this->prefix[0] = 0;
    for (int i = 0; i < n; ++i)
        this->prefix[i] = this->prefix[i - 1] + c_ga->block_num[i - 1];
    for (int i = 0; i < n; i += this->max_batch){
        int pn = std::min(this->max_batch, n-i);
        this->cq_mm->mul(x, c_ga->cq.data(), c_ga->cq_offset.data(), this->max_batch, this->cq_num, &cq_res);
        
        r_dot_prod<T>(x, c_ga->pq.data(), c_ga->pq_offset.data(), pn * this->code_len, this->pq_num, this->pq_dimension, this->code_book, this->pq_num);

        for (int j = 0; j < pn; ++j){
            int cnt = 0;
            for (int _j = 0; _j < this->select_cq; ++_j){
                int cq_id = cq_res[cq_id].second;
                uint8_t* data= c_ga->data[cq_id].data();
                int num = c_ga->block_num[cq_id];
                for (int vec_id = 0; vec_id < num; vec_id += this->res_cache_size){
                    int qn = std::min(this->max_block, num - vec_id);
                    if (cnt + qn >= this->res_cache_size){
                        std::nth_element(this->res, this->res + this->topk + 1, this->res + cnt + 1, pair_greator<Tout, int>());
                        for (int k = 0; k < this->topk; ++k)
                        ans.push_back(res[k]);
                        memset(this->res, 0, sizeof(pair<Tout, int>) * cnt);
                        cnt = 0;
                    }

                    get_res<Tout>(data + vec_id * this->dimension, &this->code_book[j * this->code_len * this->pq_num],
                                  this->code_len, this->pq_num, cnt, this->prefix[cq_id], qn, &this->res[(i + j) * this->max_block + vec_id]);

                    cnt += qn;
                    
                }
            }
            std::nth_element(ans.data(), ans.data() + this->topk + 1, ans.data() + ans.size() + 1, pair_greator<Tout, int>());
            std::sort(ans.data(), ans.data() + this->topk + 1, pair_greator<Tout, int>());
            for (int k =0; k < this->topk; ++k){
                sims[(i + j) * this->topk + k] = ans[k].first;
                int v = std::lower_bound(this->prefix, this->prefix + this->cq_num + 1, ans[k].second) - this->prefix;
                idx[(i + j) * this->topk + k] = c_ga->ids[v][ans[k].second - this->prefix[v]];
            }
            ans.clear();
        }
    }
    return 0;
}

template<typename T,
        DistanceType dist_type>
int pqivf_probe<T, dist_type>::query_with_uids(const T* const x, const int n, gallery<T, dist_type> * ga, uint32_t *uids, const int m, Tout *sims, uint32_t *idx){
    /* TODO */
    return NO_SUPPORT;
}

}
