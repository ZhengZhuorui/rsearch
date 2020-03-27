#include "probe/pqivf_mt_probe.h"
namespace rsearch{
template<typename T,
        DistanceType dist_type>
pqivf_mt_probe<T, dist_type>::pqivf_mt_probe(int dimension, int topk):probe<T, dist_type>(){
    
    this->dimension = dimension;
    this->topk = topk;
    this->nprocs = std::thread::hardware_concurrency();
    this->max_batch = 32;
    this->max_block = 102400;
    this->cq_num = 4096;
    this->select_cq = 2048;
    this->res_cache_size = 102400;
    this->cq_mm = new rapid_matrix_mul<T>;
    this->cq_mm->set(this->dimension, this->select_cq, max_batch, max_block);
    this->code_len = this->dimension / this->pq_dimension;
    this->res = (pair<Tout, idx_t>*)malloc(this->res_cache_size * sizeof(pair<Tout, idx_t>*));
    memset(this->res, 0, this->res_cache_size * sizeof(pair<Tout, idx_t>));
    this->code_book = (Tout*)malloc(this->max_batch * this->pq_num * this->code_len * sizeof(Tout));
    this->prefix = (int32_t*)malloc((this->cq_num + 5) * sizeof(int32_t));
    this->thread_pool
}
template pqivf_mt_probe<int8_t, COSINE>::pqivf_mt_probe(int, int);
template pqivf_mt_probe<float, COSINE>::pqivf_mt_probe(int, int);
template pqivf_mt_probe<int8_t, EUCLIDEAN>::pqivf_mt_probe(int, int);
template pqivf_mt_probe<float, EUCLIDEAN>::pqivf_mt_probe(int, int);

template<typename T,
        DistanceType dist_type>
pqivf_mt_probe<T, dist_type>::~pqivf_mt_probe(){
    delete this->cq_mm;
    //delete this->pq_mm;
    free(this->res);
    free(this->prefix);
    free(this->code_book);
}
template pqivf_mt_probe<int8_t, COSINE>::~pqivf_mt_probe();
template pqivf_mt_probe<float, COSINE>::~pqivf_mt_probe();
template pqivf_mt_probe<int8_t, EUCLIDEAN>::~pqivf_mt_probe();
template pqivf_mt_probe<float, EUCLIDEAN>::~pqivf_mt_probe();

template<typename T,
        DistanceType dist_type>
int pqivf_mt_probe<T, dist_type>::create_gallery(gallery<T, dist_type> ** ga_ptr){
    struct pqivf_traits traits = {4096,2048,32,256};
    pqivf_gallery<T, dist_type> * ga = new pqivf_gallery<T, dist_type>(this->dimension, traits);
    (*ga_ptr) = (gallery<T, dist_type>*)ga;
    return 0;
}
template int pqivf_mt_probe<int8_t, COSINE>::create_gallery(gallery<int8_t, COSINE> ** ga_ptr);
template int pqivf_mt_probe<float, COSINE>::create_gallery(gallery<float, COSINE> ** ga_ptr);
template int pqivf_mt_probe<int8_t, EUCLIDEAN>::create_gallery(gallery<int8_t, EUCLIDEAN> ** ga_ptr);
template int pqivf_mt_probe<float, EUCLIDEAN>::create_gallery(gallery<float, EUCLIDEAN> ** ga_ptr);

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
int pqivf_mt_probe<T, dist_type>::query(const T * const x, const int n, gallery<T, dist_type> * ga, Tout *sims, uint32_t *idx){

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
    vector<T> x_tmp(n * this->dimension);
    memcpy(x_tmp.data(), x, n * this->dimension * sizeof(T));
    if (is_same_type<T, int8_t>() == true){
        for (int i = 0; i < n * this->dimension; ++i)
            x_tmp[i] += 64;
    }

    pair<Tout, idx_t>* cq_res;
    //pair<Tout, idx_t>* pq_res;
    this->prefix[0] = 0;
    for (int i = 0; i < n; ++i)
        this->prefix[i] = this->prefix[i - 1] + c_ga->block_num[i - 1];
    for (int i = 0; i < n; i += this->max_batch){
        int pn = std::min(this->max_batch, n-i);
        this->cq_mm->mul(x_tmp.data(), c_ga->cq.data(), c_ga->cq_offset.data(), this->max_batch, this->cq_num, &cq_res);
        
        r_dot_prod<T>(x_tmp.data(), c_ga->pq.data(), c_ga->pq_offset.data(), pn * this->code_len, this->pq_num, this->pq_dimension, this->code_book, this->pq_num);

        for (int j = 0; j < pn; ++j){
            int cnt = 0;
            for (int _j = 0; _j < this->select_cq; ++_j){
                int cq_id = cq_res[j * this->select_cq + _j].second;
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
                //sims[(i + j) * this->topk + k] = vec_dis<T, dist_type>(x + 1LL * (i + j) * this->dimension, c_ga->data[v].data() + ans[k].second - this->prefix[v], this->dimension);
            }
            ans.clear();
        }
    }
    return 0;
}
template int pqivf_mt_probe<int8_t, COSINE>::query(const int8_t * const x, const int n, gallery<int8_t, COSINE> * ga, int *sims, uint32_t *idx);
template int pqivf_mt_probe<float, COSINE>::query(const float * const x, const int n, gallery<float, COSINE> * ga, float *sims, uint32_t *idx);
template int pqivf_mt_probe<int8_t, EUCLIDEAN>::query(const int8_t * const x, const int n, gallery<int8_t, EUCLIDEAN> * ga, int *sims, uint32_t *idx);
template int pqivf_mt_probe<float, EUCLIDEAN>::query(const float * const x, const int n, gallery<float, EUCLIDEAN> * ga, float *sims, uint32_t *idx);

template<typename T,
        DistanceType dist_type>
int pqivf_mt_probe<T, dist_type>::query_with_uids(const T* const x, const int n, gallery<T, dist_type> * ga, uint32_t *uids, const int m, Tout *sims, uint32_t *idx){
    /* TODO */
    return NO_SUPPORT;
}
template int pqivf_mt_probe<int8_t, COSINE>::query_with_uids(const int8_t * const x, const int n, gallery<int8_t, COSINE> * ga, uint32_t *uids, const int m, int *sims, uint32_t *idx);
template int pqivf_mt_probe<float, COSINE>::query_with_uids(const float * const x, const int n, gallery<float, COSINE> * ga, uint32_t *uids, const int m, float *sims, uint32_t *idx);
template int pqivf_mt_probe<int8_t, EUCLIDEAN>::query_with_uids(const int8_t * const x, const int n, gallery<int8_t, EUCLIDEAN> * ga, uint32_t *uids, const int m,  int *sims, uint32_t *idx);
template int pqivf_mt_probe<float, EUCLIDEAN>::query_with_uids(const float * const x, const int n, gallery<float, EUCLIDEAN> * ga, uint32_t *uids, const int m, float *sims, uint32_t *idx);

}
