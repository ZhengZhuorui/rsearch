#include "probe/cpu_base_mt_probe.h"
#include <sys/time.h>
namespace rsearch{
template<typename T,
        DistanceType dist_type,
        typename matrix_type>
cpu_base_mt_probe<T, dist_type, matrix_type>::cpu_base_mt_probe(int dimension, int topk) : base_probe<T, dist_type, matrix_type>(){
    this->nprocs = std::thread::hardware_concurrency();
    this->mm.resize(this->nprocs);
    this->dimension = dimension;
    this->topk = topk;
    this->max_batch = 32;
    this->max_block = 102400;
    this->x_tmp.resize(this->max_batch * this->dimension);
    for (int i = 0; i < this->nprocs; ++i){
        this->mm[i] = new matrix_type;
        this->mm[i]->set(this->dimension, this->topk, this->max_batch, this->max_block);
    }
    ans.resize(this->max_batch);
    //for (int i = 0; i < max_batch; ++i)
    //    ans.clear();
    this->threadpool = new ThreadPool;
    this->mth_manager = new MthManager;

}

template cpu_base_mt_probe<int8_t, COSINE, rapid_matrix_mul<int8_t> >::cpu_base_mt_probe(int, int);
template cpu_base_mt_probe<float, COSINE, rapid_matrix_mul<float> >::cpu_base_mt_probe(int, int);
template cpu_base_mt_probe<int8_t, EUCLIDEAN, rapid_matrix_mul<int8_t> >::cpu_base_mt_probe(int, int);
template cpu_base_mt_probe<float, EUCLIDEAN, rapid_matrix_mul<float> >::cpu_base_mt_probe(int, int);

template<typename T,
        DistanceType dist_type,
        typename matrix_type>
cpu_base_mt_probe<T, dist_type, matrix_type>::~cpu_base_mt_probe(){
    for (int i = 0; i < this->nprocs; ++i){
        delete this->mm[i];
    }
}

template cpu_base_mt_probe<int8_t, COSINE, rapid_matrix_mul<int8_t> >::~cpu_base_mt_probe();
template cpu_base_mt_probe<float, COSINE, rapid_matrix_mul<float> >::~cpu_base_mt_probe();
template cpu_base_mt_probe<int8_t, EUCLIDEAN, rapid_matrix_mul<int8_t> >::~cpu_base_mt_probe();
template cpu_base_mt_probe<float, EUCLIDEAN, rapid_matrix_mul<float> >::~cpu_base_mt_probe();

template<typename T,
        DistanceType dist_type,
        typename matrix_type>
int cpu_base_mt_probe<T, dist_type, matrix_type>::create_gallery(gallery<T> ** ga_ptr){
    cpu_base_gallery<T, dist_type> * ga = new cpu_base_gallery<T, dist_type>(this->dimension);
    //int ret = ga->init();
    (*ga_ptr) = (gallery<T>*)ga;
    return 0;
}

template int cpu_base_mt_probe<int8_t, COSINE, rapid_matrix_mul<int8_t> >::create_gallery(gallery<int8_t> ** ga_ptr);
template int cpu_base_mt_probe<float, COSINE, rapid_matrix_mul<float> >::create_gallery(gallery<float> ** ga_ptr);
template int cpu_base_mt_probe<int8_t, EUCLIDEAN, rapid_matrix_mul<int8_t> >::create_gallery(gallery<int8_t> ** ga_ptr);
template int cpu_base_mt_probe<float, EUCLIDEAN, rapid_matrix_mul<float> >::create_gallery(gallery<float> ** ga_ptr);

template<typename T,
        DistanceType dist_type,
        typename matrix_type>
void cpu_base_mt_probe<T, dist_type, matrix_type>::query_bunch(const int thread_id, const T* x, const T* data, const Tout* offset, const int batch, const int block, const int block_id){    
    pair<Tout, idx_t>* res;
    this->mm[thread_id]->mul(x, data, offset, batch, block, &res);
    int base_id = block_id * this->max_block;
    for (int k = 0; k < batch; ++k){
        for (int l = 0; l < this->topk; ++l)
            this->ans[k][block_id * this->topk + l] = std::make_pair(res[k * this->topk + l].first, res[k * this->topk + l].second + base_id);
    }
}

template void cpu_base_mt_probe<int8_t, COSINE, rapid_matrix_mul<int8_t> >::query_bunch(const int mm_id, const int8_t* x, const int8_t* data, const int* offset, const int batch, const int block, const int base_id);
template void cpu_base_mt_probe<float, COSINE, rapid_matrix_mul<float> >::query_bunch(const int mm_id, const float* x, const float* data, const float* offset, const int batch, const int block, const int base_id);
template void cpu_base_mt_probe<int8_t, EUCLIDEAN, rapid_matrix_mul<int8_t>>::query_bunch(const int mm_id, const int8_t* x, const int8_t* data, const int* offset, const int batch, const int block, const int base_id);
template void cpu_base_mt_probe<float, EUCLIDEAN, rapid_matrix_mul<float> >::query_bunch(const int mm_id, const float* x, const float* data, const float* offset, const int batch, const int block, const int base_id);

template<typename T,
        DistanceType dist_type,
        typename matrix_type>
int cpu_base_mt_probe<T, dist_type, matrix_type>::query(const T * const x, const int n, gallery<T> * ga, Tout *sims, idx_t *idx){
    cpu_base_gallery<T, dist_type>* c_ga = (cpu_base_gallery<T, dist_type>*) ga;
    int num = c_ga->num;
    T* data= (T*)c_ga->data.data();
    Tout* offset = (Tout*)c_ga->offset.data();

    struct timezone zone;
    struct timeval time1;
    struct timeval time2;
    float delta = 0;
    if (num < this->topk){
        //std::cout<< "num < topk" <<num<< std::endl;
        ans.resize(n);
        for (int i = 0; i < n; ++i){
            for (int j = 0; j < num; ++j)
                ans[i].push_back(std::make_pair(vec_dis<T, dist_type>(x + 1LL * i * this->dimension, data + 1LL * j * this->dimension, this->dimension), j));
            sort(ans[i].begin(), ans[i].end(), pair_greator<T, idx_t>());
            for (int j = 0; j < num; ++j){
                sims[i * this->topk + j] = ans[i][j].first;
                idx[i * this->topk + j] = c_ga->ids[ans[i][j].second];
            }
            for (int j = num; j < this->topk; ++j){
                sims[i * this->topk + j] = 0;
                idx[i * this->topk + j] = -1;
            }
        }
        return 0;
    }
    this->ans_topk_size = ((num - 1) / this->max_block + 1) * topk;
    std::cout << std::thread::hardware_concurrency() << std::endl;
    for (int i = 0; i < this->max_batch; ++i)
        this->ans[i].resize(this->ans_topk_size);
    this->threadpool->start();
    for (int i = 0 ; i < n ; i += this->max_batch){
        int pn = std::min(this->max_batch, n - i);
        memcpy(this->x_tmp.data(), x + 1LL * i * this->dimension, pn * this->dimension * sizeof(T));
        if (is_same_type<T, int8_t>() == true){
            for (int64_t k = 0; k < 1LL * pn * this->dimension; ++k){
                this->x_tmp[k] += 64;
            }
        }
        std::cout << "[query] target 1" << std::endl;
        for (int j = 0; j < num ; j += this->max_block){
            int block_size = std::min(this->max_block, num - j);
            //this->query_bunch(mm_id, x_tmp.data(), data + 1LL * j * this->dimension, offset + j, pn, block_size, j);
            std::function<void(int)> f = std::bind(&cpu_base_mt_probe<T, dist_type, matrix_type>::query_bunch, this, std::placeholders::_1, x_tmp.data(),
                                                 data + 1LL * j * this->dimension, offset + j, pn, block_size, j / this->max_block);
            this->mth_manager->add_task(f);
        }
        std::cout << "[query] target 2 " << this->mth_manager->size() << std::endl;
        std::function<void()> work = std::bind(&MthManager::work, this->mth_manager);
        int work_sz = this->mth_manager->size();

        gettimeofday(&time1, &zone);
        for (int i = 0; i < work_sz; ++i)
            this->threadpool->add_task(work);

        this->threadpool->synchronize();
        gettimeofday(&time2, &zone);
        delta += (time2.tv_sec - time1.tv_sec) * 1000.0 + (time2.tv_usec - time1.tv_usec) / 1000.0;

        std::cout << "[query] target 3 " << " " << this->mth_manager->size() << std::endl;
        for (int k = 0; k < pn; ++k){
            //std::cout << c_ga->offset[ans[k][0].second] << " ";

            std::nth_element(ans[k].data(), ans[k].data() + this->topk + 1, ans[k].data() + ans[k].size() + 1,
                             pair_greator<Tout, idx_t>());
            std::sort(ans[k].data(), ans[k].data() + this->topk + 1, pair_greator<Tout, idx_t>());
            for (int j = 0 ; j < this->topk ; ++j){
                sims[(i + k) * this->topk + j] = ans[k][j].first;
                idx[(i + k) * this->topk + j] = c_ga->ids[ans[k][j].second];
            }
            memset(ans[k].data(), 0, this->ans_topk_size * sizeof(pair<Tout, idx_t>));
        }
    }
    this->threadpool->stop();
    printf("[query] cost %.4fms.\n", delta);
    return 0;
}

template int cpu_base_mt_probe<int8_t, COSINE, rapid_matrix_mul<int8_t> >::query(const int8_t * const x, const int n, gallery<int8_t> * ga, int *sims, idx_t *idx);
template int cpu_base_mt_probe<float, COSINE, rapid_matrix_mul<float> >::query(const float * const x, const int n, gallery<float> * ga, float *sims, idx_t *idx);
template int cpu_base_mt_probe<int8_t, EUCLIDEAN, rapid_matrix_mul<int8_t>>::query(const int8_t * const x, const int n, gallery<int8_t> * ga, int *sims, idx_t *idx);
template int cpu_base_mt_probe<float, EUCLIDEAN, rapid_matrix_mul<float> >::query(const float * const x, const int n, gallery<float> * ga, float *sims, idx_t *idx);

template<typename T,
        DistanceType dist_type,
        typename matrix_type>
int cpu_base_mt_probe<T, dist_type, matrix_type>::query_with_uids(const T* const x, const int n, gallery<T> * ga, idx_t *uids, const int m, Tout *sims, idx_t *idx){
    
    /* TODO */
    
    return NO_SUPPORT;
}

template int cpu_base_mt_probe<int8_t, COSINE, rapid_matrix_mul<int8_t> >::query_with_uids(const int8_t * const x, const int n, gallery<int8_t> * ga, idx_t *uids, const int m, int *sims, idx_t *idx);
template int cpu_base_mt_probe<float, COSINE, rapid_matrix_mul<float> >::query_with_uids(const float * const x, const int n, gallery<float> * ga, idx_t *uids, const int m, float *sims, idx_t *idx);
template int cpu_base_mt_probe<int8_t, EUCLIDEAN, rapid_matrix_mul<int8_t>>::query_with_uids(const int8_t * const x, const int n, gallery<int8_t> * ga, idx_t *uids, const int m, int *sims, idx_t *idx);
template int cpu_base_mt_probe<float, EUCLIDEAN, rapid_matrix_mul<float> >::query_with_uids(const float * const x, const int n, gallery<float> * ga, idx_t *uids, const int m, float *sims, idx_t *idx);
}
