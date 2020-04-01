#include "probe/pqivf_probe.h"
namespace rsearch{
template<typename T,
        DistanceType dist_type>
pqivf_probe<T, dist_type>::pqivf_probe(int dimension, int topk):probe<T, dist_type>(){
    this->cq_mm = new rapid_matrix_mul<T>;
    this->mtx_la = new rapid_matrix_la<Tout>;

    this->dimension = dimension;
    this->topk = topk;
    this->max_batch = 32;
    this->max_block = 102400;

    this->cq_num = 4096;
    this->select_cq = 2048;
    this->pq_num = 256;
    this->pq_dimension = 32;
    this->code_len = this->dimension / this->pq_dimension;
    this->res_cache_size = 102400;
    this->res = (pair<Tout, idx_t>*)malloc(this->res_cache_size * sizeof(pair<Tout, idx_t>));
    memset(this->res, 0, this->res_cache_size * sizeof(pair<Tout, idx_t>));
    this->code_book = (Tout*)malloc(this->max_batch * this->pq_num * this->code_len * sizeof(Tout));
    this->prefix = (int32_t*)malloc((this->cq_num + 5) * sizeof(int32_t));
    this->nprocs = std::thread::hardware_concurrency();

    this->x_tmp.resize(this->max_batch * this->dimension);
    this->cq_mm->set(this->dimension, this->select_cq, this->max_batch, this->cq_num);
    this->mtx_la->set(this->code_len, this->topk, 1, this->max_block, this->pq_num);

}
template pqivf_probe<int8_t, COSINE>::pqivf_probe(int, int);
template pqivf_probe<float, COSINE>::pqivf_probe(int, int);
template pqivf_probe<int8_t, EUCLIDEAN>::pqivf_probe(int, int);
template pqivf_probe<float, EUCLIDEAN>::pqivf_probe(int, int);

template<typename T,
        DistanceType dist_type>
pqivf_probe<T, dist_type>::~pqivf_probe(){
    delete this->cq_mm;
    free(this->res);
    free(this->prefix);
    free(this->code_book);
}
template pqivf_probe<int8_t, COSINE>::~pqivf_probe();
template pqivf_probe<float, COSINE>::~pqivf_probe();
template pqivf_probe<int8_t, EUCLIDEAN>::~pqivf_probe();
template pqivf_probe<float, EUCLIDEAN>::~pqivf_probe();

template<typename T,
        DistanceType dist_type>
int pqivf_probe<T, dist_type>::create_gallery(gallery<T, dist_type> ** ga_ptr){
    struct pqivf_traits traits = {4096,2048,32,256};
    pqivf_gallery<T, dist_type> * ga = new pqivf_gallery<T, dist_type>(this->dimension, traits);
    (*ga_ptr) = (gallery<T, dist_type>*)ga;
    return 0;
}
template int pqivf_probe<int8_t, COSINE>::create_gallery(gallery<int8_t, COSINE> ** ga_ptr);
template int pqivf_probe<float, COSINE>::create_gallery(gallery<float, COSINE> ** ga_ptr);
template int pqivf_probe<int8_t, EUCLIDEAN>::create_gallery(gallery<int8_t, EUCLIDEAN> ** ga_ptr);
template int pqivf_probe<float, EUCLIDEAN>::create_gallery(gallery<float, EUCLIDEAN> ** ga_ptr);

template<typename T>
inline void get_res(int* data, int* code_book, int code_len, int ldc, int qid, int block, pair<T, idx_t>* res){
    /*
        for (int i = 0; i < block; ++i)
            for (int j = 0; j < code_len; ++j){
                res[i].first = code_book[j * pq_num + data[i * code_len + j]];
                res[i].second = qid + i;
            }
    */

    for (int i = 0, _i = 0; i < block; ++i, _i += code_len)
        for (int j = 0, _j = 0; j < code_len; ++j, _j += ldc)
            res[i].first += code_book[_j + data[_i + j]];

    for (int i = 0; i < block; ++i) 
        res[i].second = qid + i;
}

template<typename T,
        DistanceType dist_type>
int pqivf_probe<T, dist_type>::query(const T * const x, const int n, gallery<T, dist_type> * ga, Tout *sims, idx_t *idx){

    pqivf_gallery<T, dist_type>* c_ga = (pqivf_gallery<T, dist_type>*)ga;
    vector<pair<Tout, idx_t> > ans;
    c_ga->mtx.lock();
    std::cout << "[query] target 1 : " << this->pq_num << std::endl;
    vector<T> x_tmp(this->max_batch * this->dimension);

    pair<Tout, idx_t>* cq_res;
    this->prefix[0] = 0;
    for (int i = 1; i <= this->cq_num; ++i){
        this->prefix[i] = this->prefix[i - 1] + c_ga->block_num[i - 1];
    }
    std::cout << std::endl;
    //memset(this->res, 0, this->res_cache_size * sizeof(pair<Tout, idx_t>));
    pair<Tout, idx_t>* mtx_res;
    std::cout << "[query] target 2 : " << this->pq_num << " " << this->prefix[this->cq_num]<< std::endl;
    for (int i = 0; i < n; i += this->max_batch){
        int pn = std::min(this->max_batch, n - i);
        
        memcpy(x_tmp.data(), x + 1LL * i * this->dimension, 1LL * pn * this->dimension * sizeof(T));

        if (is_same_type<T, int8_t>() == true){
            for (int64_t k = 0; k < 1LL * pn * this->dimension; ++k)
                x_tmp[k] += 64;
        }
        this->cq_mm->mul(x_tmp.data(), c_ga->cq.data(), c_ga->cq_offset.data(), pn, this->cq_num, &cq_res);
        
        r_dot_prod<T>(x_tmp.data(), c_ga->pq.data(), c_ga->pq_offset.data(), pn * this->code_len, this->pq_num, this->pq_dimension, this->code_book, this->pq_num);
        //for (int j = 0; j < this->pq_num; ++j)
        //    std::cout << this->code_book[j] << "?";

        for (int j = 0; j < pn; ++j){
            //int cnt = 0;
            for (int _j = 0; _j < this->select_cq; ++_j){

                int cq_id = cq_res[j * this->select_cq + _j].second;
                int* data= c_ga->data[cq_id].data();
                int num = c_ga->block_num[cq_id];
                for (int vec_id = 0; vec_id < num; vec_id += this->res_cache_size){
                    int qn = std::min(this->max_block, num - vec_id);
                    /*if (cnt + qn > this->res_cache_size){
                        if (cnt > this->topk + 1){
                            std::nth_element(this->res, this->res + this->topk + 1, this->res + cnt + 1, pair_greator<Tout, int>());
                            for (int k = 0; k < this->topk; ++k)
                                ans.push_back(res[k]);
                        }
                        else{
                            for (int k = 0; k < cnt; ++k)
                                ans.push_back(res[k]);
                        }
                        memset(this->res, 0, sizeof(pair<Tout, int>) * cnt);
                        cnt = 0;
                    }*/
                    //std::cout << "[query] target 4.1" << std::endl;
                    //get_res(data + 1LL * vec_id * this->dimension, &this->code_book[j * this->code_len * this->pq_num],
                    //              this->code_len, this->pq_num, this->prefix[cq_id], qn, this->res + cnt);
                    this->mtx_la->la(data + vec_id, code_book, 1, qn, &mtx_res);
                    int k_sz = std::min(qn, this->topk);
                    for (int k = 0; k < k_sz; ++k)
                        ans.push_back(std::make_pair(mtx_res[k].first, mtx_res[k].second + this->prefix[cq_id] + vec_id));
                }
            }
            /*
            if (cnt > this->topk + 1){
                std::nth_element(this->res, this->res + this->topk + 1, this->res + cnt + 1, pair_greator<Tout, int>());
                for (int k = 0; k < this->topk; ++k)
                    ans.push_back(res[k]);
            }
            else{
                for (int k = 0; k < cnt; ++k)
                    ans.push_back(res[k]);
            }
            memset(this->res, 0, sizeof(pair<Tout, int>) * cnt);
            cnt = 0;*/
            
            //std::cout << "target 6 " << ans.size() << " " << this->prefix[this->cq_num] << std::endl;
            if ((int32_t)ans.size() > this->topk){
                std::nth_element(ans.data(), ans.data() + this->topk + 1, ans.data() + ans.size() + 1, pair_greator<Tout, int>());
                std::sort(ans.data(), ans.data() + this->topk + 1, pair_greator<Tout, int>());
                for (int k =0; k < this->topk; ++k){
                    sims[(i + j) * this->topk + k] = ans[k].first;
                    //std::cout << this->prefix[0] << " " << this->prefix[this->cq_num] << " | " << ans[k].second << std::endl;
                    int v = std::upper_bound(this->prefix, this->prefix + this->cq_num + 1, ans[k].second) - this->prefix - 1;
                    idx[(i + j) * this->topk + k] = c_ga->ids[v][ans[k].second - this->prefix[v]];
                }
                ans.clear();
            }
            else {
                std::sort(ans.begin(), ans.end(), pair_greator<Tout, int>());
                for (uint32_t k =0; k < ans.size(); ++k){
                    sims[(i + j) * this->topk + k] = ans[k].first;
                    int v = std::upper_bound(this->prefix, this->prefix + this->cq_num + 1, ans[k].second) - this->prefix - 1;
                    idx[(i + j) * this->topk + k] = c_ga->ids[v][ans[k].second - this->prefix[v]];
                }
                for (int k = ans.size(); k < this->topk; ++k){
                    sims[(i + j) * this->topk + k] = 0;
                    idx[(i + j) * this->topk + k] = -1;
                }
                ans.clear();
            }
        }
    }
    return 0;
}
template int pqivf_probe<int8_t, COSINE>::query(const int8_t * const x, const int n, gallery<int8_t, COSINE> * ga, int *sims, idx_t *idx);
template int pqivf_probe<float, COSINE>::query(const float * const x, const int n, gallery<float, COSINE> * ga, float *sims, idx_t *idx);
template int pqivf_probe<int8_t, EUCLIDEAN>::query(const int8_t * const x, const int n, gallery<int8_t, EUCLIDEAN> * ga, int *sims, idx_t *idx);
template int pqivf_probe<float, EUCLIDEAN>::query(const float * const x, const int n, gallery<float, EUCLIDEAN> * ga, float *sims, idx_t *idx);

template<typename T,
        DistanceType dist_type>
int pqivf_probe<T, dist_type>::query_with_uids(const T* const x, const int n, gallery<T, dist_type> * ga, idx_t *uids, const int m, Tout *sims, idx_t *idx){
    /* TODO */
    return NO_SUPPORT;
}
template int pqivf_probe<int8_t, COSINE>::query_with_uids(const int8_t * const x, const int n, gallery<int8_t, COSINE> * ga, idx_t *uids, const int m, int *sims, idx_t *idx);
template int pqivf_probe<float, COSINE>::query_with_uids(const float * const x, const int n, gallery<float, COSINE> * ga, idx_t *uids, const int m, float *sims, idx_t *idx);
template int pqivf_probe<int8_t, EUCLIDEAN>::query_with_uids(const int8_t * const x, const int n, gallery<int8_t, EUCLIDEAN> * ga, idx_t *uids, const int m,  int *sims, idx_t *idx);
template int pqivf_probe<float, EUCLIDEAN>::query_with_uids(const float * const x, const int n, gallery<float, EUCLIDEAN> * ga, idx_t *uids, const int m, float *sims, idx_t *idx);

}
