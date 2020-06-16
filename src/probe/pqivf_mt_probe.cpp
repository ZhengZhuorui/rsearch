#include "probe/pqivf_mt_probe.h"
#include <sys/time.h>
#define INF 1000000000
namespace rsearch{
template<typename T,
        DistanceType dist_type>
pqivf_mt_probe<T, dist_type>::pqivf_mt_probe(int dimension, int topk):probe<T>(){

    this->nprocs = std::thread::hardware_concurrency();
    this->cq_mm = new rapid_matrix_mul<T>;
    this->mtx_la.resize(nprocs);
    for (int i = 0; i < this->nprocs; ++i)
        this->mtx_la[i] = new rapid_matrix_la<Tout>;
    this->dimension = dimension;
    this->topk = topk;
    this->max_batch = 32;
    this->max_block = 10240;

    this->cq_num = 4096;
    this->select_cq = 1024;
    this->pq_num = 256;
    this->pq_dimension = 16;
    this->code_len = this->dimension / this->pq_dimension;
    this->codebook_size = this->code_len * this->pq_num;

    //this->res = (pair<Tout, idx_t>*)malloc(this->res_cache_size * sizeof(pair<Tout, idx_t>));
    //memset(this->res, 0, this->res_cache_size * sizeof(pair<Tout, idx_t>));
    this->code_book = (Tout*)malloc(this->max_batch * this->codebook_size * sizeof(Tout));
    this->prefix = (int*)malloc((this->cq_num + 5) * sizeof(int));

    this->x_tmp.resize(this->max_batch * this->dimension);
    this->x_tmp_div.resize(this->max_batch * this->dimension);
    this->x_offset.resize(this->max_batch);
    this->res_cache_size = this->nprocs * this->topk;
    this->ans.resize(this->max_batch * this->res_cache_size);
    
    //for (int i = 0; i < this->max_batch * this->res_cache_size; ++i)
        //this->ans[i] = std::make_pair(-INF, -1);
    memset(this->ans.data(), -1, 1LL * this->max_batch * this->res_cache_size * sizeof(pair<Tout, idx_t>));
    this->merge_cache = (pair<Tout, idx_t>*)malloc(this->res_cache_size * sizeof(pair<Tout, idx_t>));

    //for (int i = 0; i < this->max_batch; ++i)
    //    ans[i].resize(this->mm_id * this->topk);
    

    this->cq_mm->set(this->dimension, this->select_cq, this->max_batch, this->cq_num);
    for (int i = 0; i < this->nprocs; ++i)
        this->mtx_la[i]->set(this->code_len, this->topk, 1, this->max_block, this->codebook_size);

    this->threadpool = new ThreadPool;
    this->mth_manager = new MthManager;

}
template pqivf_mt_probe<int8_t, COSINE>::pqivf_mt_probe(int, int);
template pqivf_mt_probe<float, COSINE>::pqivf_mt_probe(int, int);
template pqivf_mt_probe<int8_t, EUCLIDEAN>::pqivf_mt_probe(int, int);
template pqivf_mt_probe<float, EUCLIDEAN>::pqivf_mt_probe(int, int);

template<typename T,
        DistanceType dist_type>
pqivf_mt_probe<T, dist_type>::~pqivf_mt_probe(){
    delete this->cq_mm;
    delete this->threadpool;
    delete this->mth_manager;
    //free(this->res);
    free(this->prefix);
    free(this->code_book);
    free(this->merge_cache);
}
template pqivf_mt_probe<int8_t, COSINE>::~pqivf_mt_probe();
template pqivf_mt_probe<float, COSINE>::~pqivf_mt_probe();
template pqivf_mt_probe<int8_t, EUCLIDEAN>::~pqivf_mt_probe();
template pqivf_mt_probe<float, EUCLIDEAN>::~pqivf_mt_probe();

template<typename T,
        DistanceType dist_type>
int pqivf_mt_probe<T, dist_type>::create_gallery(gallery<T> ** ga_ptr){
    struct pqivf_traits traits = {this->cq_num, this->select_cq, this->pq_dimension, this->pq_num};
    pqivf_gallery<T, dist_type> * ga = new pqivf_gallery<T, dist_type>(this->dimension, traits);
    (*ga_ptr) = (gallery<T>*)ga;
    return 0;
}
template int pqivf_mt_probe<int8_t, COSINE>::create_gallery(gallery<int8_t> ** ga_ptr);
template int pqivf_mt_probe<float, COSINE>::create_gallery(gallery<float> ** ga_ptr);
template int pqivf_mt_probe<int8_t, EUCLIDEAN>::create_gallery(gallery<int8_t> ** ga_ptr);
template int pqivf_mt_probe<float, EUCLIDEAN>::create_gallery(gallery<float> ** ga_ptr);

template<typename T>
inline void get_res(int* data, T* code_book, int code_len, int ldc, int qid, int block, pair<T, idx_t>* res){
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
void pqivf_mt_probe<T, dist_type>::query_bunch(const int thread_id, const int* data, const Tout* code_book, const int block, const int id, const int offset){
    //std::cout << "<" << thread_id  << ">" << std::endl;
    pair<Tout, idx_t>* mtx_res;
    this->mtx_la[thread_id]->la(data, code_book, 1, block, &mtx_res);
    int sz = std::min(block, this->topk);
    for (int i = 0; i < sz; ++i)
        mtx_res[i].second += offset;
    
    merge(this->merge_cache + thread_id * this->topk, this->ans.data() + id * this->res_cache_size + thread_id * this->topk, mtx_res, this->topk, sz);
}

template void pqivf_mt_probe<float, COSINE>::query_bunch(const int thread_id, const int* data, const float* code_book, const int block, const int id, const int offset);
template void pqivf_mt_probe<int8_t, COSINE>::query_bunch(const int thread_id, const int* data, const int* code_book, const int block, const int id, const int offset);
template void pqivf_mt_probe<float, EUCLIDEAN>::query_bunch(const int thread_id, const int* data, const float* code_book, const int block, const int id, const int offset);
template void pqivf_mt_probe<int8_t, EUCLIDEAN>::query_bunch(const int thread_id, const int* data, const int* code_book, const int block, const int id, const int offset);



template<typename T,
        DistanceType dist_type>
int pqivf_mt_probe<T, dist_type>::query(const T * const x, const int n, gallery<T> * ga, Tout *sims, idx_t *idx){

    pqivf_gallery<T, dist_type>* c_ga = (pqivf_gallery<T, dist_type>*)ga;
    
    
    vector<T> x_tmp(this->max_batch * this->dimension);

    pair<Tout, idx_t>* cq_res;
    this->prefix[0] = 0;
    for (int i = 1; i <= this->cq_num; ++i){
        this->prefix[i] = this->prefix[i - 1] + c_ga->block_num[i - 1];
    }
    struct timezone zone;
    struct timeval time1;
    struct timeval time2;
    struct timeval time3;
    struct timeval time4;
    float delta1 = 0;
    float delta2 = 0;
    //std::cout << "[query] target 2 : " << code_book[0]<< std::endl;
    //std::cout << "[query] target 2" << std::endl;
    gettimeofday(&time1, &zone);
    memset(this->x_offset.data(), 0, this->max_batch * sizeof(T));
    this->threadpool->start();
    for (int i = 0; i < n; i += this->max_batch){
        int pn = std::min(this->max_batch, n - i);
        gettimeofday(&time3, &zone);
        if (dist_type == EUCLIDEAN)
            for (int j = 0; j < pn; ++j)
                this->x_offset[j] = dot_prod(x + 1LL * (i + j) * dimension, x + 1LL * (i + j) * dimension, this->dimension);

        memcpy(x_tmp.data(), x + 1LL * i * this->dimension, 1LL * pn * this->dimension * sizeof(T));
        //std::cout << "[query] target 3" << std::endl;
        if (is_same_type<T, int8_t>() == true){
            for (int64_t k = 0; k < 1LL * pn * this->dimension; ++k)
                x_tmp[k] += 64;
        }
        divide(x_tmp.data(), x_tmp_div.data(), pn, this->dimension, this->pq_dimension);
        this->cq_mm->mul(x_tmp.data(), c_ga->cq.data(), c_ga->cq_offset.data(), pn, this->cq_num, &cq_res);
        for (int j = 0; j < code_len; ++j){
            r_dot_prod<T>(x_tmp_div.data() + j * pn * this->pq_dimension, c_ga->pq.data() + j * this->pq_num * this->pq_dimension,
                        c_ga->pq_offset.data() + j * this->pq_num,
                        pn, this->pq_num, this->pq_dimension, this->code_book + j * this->pq_num, this->codebook_size);
        }
        //std::cout << "[query] target 4" << std::endl;
        
        for (int j = 0; j < pn; ++j){
            for (int _j = 0; _j < this->select_cq; ++_j){
                int cq_id = cq_res[j * this->select_cq + _j].second;
                int* data= c_ga->data[cq_id].data();
                int num = c_ga->block_num[cq_id];

                for (int vec_id = 0; vec_id < num; vec_id += this->max_block){
                    int qn = std::min(this->max_block, num - vec_id);                   

                    std::function<void(int)> f = std::bind(&pqivf_mt_probe<T, dist_type>::query_bunch, this, std::placeholders::_1, data + vec_id,
                                                            code_book + j * this->codebook_size, qn, j, this->prefix[cq_id] + vec_id);
                    this->mth_manager->add_task(f);
                }
            }
        }
        std::function<void()> work = std::bind(&MthManager::work, this->mth_manager);
        int work_sz = this->mth_manager->size();
        std::cout << work_sz << std::endl;
        
        //std::cout << "[query] target 5" << std::endl;
        for (int j = 0; j < work_sz; ++j)
        this->threadpool->add_task(work);

        this->threadpool->synchronize();
        //std::cout << "[query] target 6" << std::endl;
        gettimeofday(&time4, &zone);
        delta2 += (time4.tv_sec - time3.tv_sec) * 1000.0 + (time4.tv_usec - time3.tv_usec) / 1000.0;
        if (dist_type == EUCLIDEAN)
            for (int j = 0; j < pn; ++j){
                std::sort(ans.data() + j *this->res_cache_size, ans.data() + (j + 1) * this->res_cache_size, pair_greator<Tout, int>());
                int k;
                for (k =0; k < this->topk && ans[j * this->res_cache_size + k].second != -1; ++k){
                    sims[1LL * (i + j) * this->topk + k] = this->x_offset[j] - ans[j * this->res_cache_size + k].first;
                    
                    int v = std::upper_bound(this->prefix, this->prefix + this->cq_num + 1, ans[j * this->res_cache_size + k].second) - this->prefix - 1;
                    idx[1LL * (i + j) * this->topk + k] = c_ga->ids[v][ans[j * this->res_cache_size + k].second - this->prefix[v]];
                    //ans[j * this->res_cache_size + k] = std::make_pair(-INF, -1);
                }
                for (; k < this->topk; ++k){
                    sims[1LL * (i + j) * this->topk + k] = 0;
                    idx[1LL * (i + j) * this->topk + k] = -1;
                }
                //for (k = 0; k < this->res_cache_size; ++k)
                //    ans[j * this->res_cache_size + k] = std::make_pair(-INF, -1);
            }
        else 
            for (int j = 0; j < pn; ++j){
                std::sort(ans.data() + j *this->res_cache_size, ans.data() + (j + 1) * this->res_cache_size, pair_greator<Tout, int>());
                int k;
                for (k =0; k < this->topk && ans[j * this->res_cache_size + k].second != -1; ++k){
                    sims[1LL * (i + j) * this->topk + k] = ans[j * this->res_cache_size + k].first;
                    
                    int v = std::upper_bound(this->prefix, this->prefix + this->cq_num + 1, ans[j * this->res_cache_size + k].second) - this->prefix - 1;
                    idx[1LL * (i + j) * this->topk + k] = c_ga->ids[v][ans[j * this->res_cache_size + k].second - this->prefix[v]];
                    //ans[j * this->res_cache_size + k] = std::make_pair(-INF, -1);
                }
                for (; k < this->topk; ++k){
                    sims[1LL * (i + j) * this->topk + k] = 0;
                    idx[1LL * (i + j) * this->topk + k] = -1;
                }
            }
        
        memset(this->ans.data(), -1, 1LL * pn * this->res_cache_size * sizeof(pair<Tout, idx_t>));
        //std::cout << ans[this->res_cache_size - 1].first<< " " <<ans[this->res_cache_size - 1].second << std::endl;
    }
    this->threadpool->stop();
    gettimeofday(&time2, &zone);
    delta1 += (time2.tv_sec - time1.tv_sec) * 1000.0 + (time2.tv_usec - time1.tv_usec) / 1000.0;
    std::cout << "res: " << delta1 << " " << delta2 << std::endl;
    return 0;
}
template int pqivf_mt_probe<int8_t, COSINE>::query(const int8_t * const x, const int n, gallery<int8_t> * ga, int *sims, idx_t *idx);
template int pqivf_mt_probe<float, COSINE>::query(const float * const x, const int n, gallery<float> * ga, float *sims, idx_t *idx);
template int pqivf_mt_probe<int8_t, EUCLIDEAN>::query(const int8_t * const x, const int n, gallery<int8_t> * ga, int *sims, idx_t *idx);
template int pqivf_mt_probe<float, EUCLIDEAN>::query(const float * const x, const int n, gallery<float> * ga, float *sims, idx_t *idx);

template<typename T,
        DistanceType dist_type>
int pqivf_mt_probe<T, dist_type>::query_with_uids(const T* const x, const int n, gallery<T> * ga, idx_t *uids, const int m, Tout *sims, idx_t *idx){
    /* TODO */
    return NO_SUPPORT;
}
template int pqivf_mt_probe<int8_t, COSINE>::query_with_uids(const int8_t * const x, const int n, gallery<int8_t> * ga, idx_t *uids, const int m, int *sims, idx_t *idx);
template int pqivf_mt_probe<float, COSINE>::query_with_uids(const float * const x, const int n, gallery<float> * ga, idx_t *uids, const int m, float *sims, idx_t *idx);
template int pqivf_mt_probe<int8_t, EUCLIDEAN>::query_with_uids(const int8_t * const x, const int n, gallery<int8_t> * ga, idx_t *uids, const int m,  int *sims, idx_t *idx);
template int pqivf_mt_probe<float, EUCLIDEAN>::query_with_uids(const float * const x, const int n, gallery<float> * ga, idx_t *uids, const int m, float *sims, idx_t *idx);

}
