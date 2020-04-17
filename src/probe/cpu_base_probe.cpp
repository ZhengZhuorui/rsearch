#include "probe/cpu_base_probe.h"
namespace rsearch{
template<typename T,
        DistanceType dist_type,
        typename matrix_type>
cpu_base_probe<T, dist_type, matrix_type>::cpu_base_probe(int dimension, int topk) : base_probe<T, dist_type, matrix_type>(){
    this->mm = new matrix_type;
    this->dimension = dimension;
    this->topk = topk;
    this->max_batch = 32;
    this->max_block = 102400;
    this->x_tmp.resize(this->max_batch * this->dimension);
    this->mm->set(this->dimension, this->topk, this->max_batch, this->max_block);
}
template cpu_base_probe<int8_t, COSINE, base_matrix_mul<int8_t> >::cpu_base_probe(int, int);
template cpu_base_probe<float, COSINE, base_matrix_mul<float> >::cpu_base_probe(int, int);
template cpu_base_probe<int8_t, EUCLIDEAN, base_matrix_mul<int8_t>>::cpu_base_probe(int, int);
template cpu_base_probe<float, EUCLIDEAN, base_matrix_mul<float> >::cpu_base_probe(int, int);

template cpu_base_probe<int8_t, COSINE, rapid_matrix_mul<int8_t> >::cpu_base_probe(int, int);
template cpu_base_probe<float, COSINE, rapid_matrix_mul<float> >::cpu_base_probe(int, int);
template cpu_base_probe<int8_t, EUCLIDEAN, rapid_matrix_mul<int8_t> >::cpu_base_probe(int, int);
template cpu_base_probe<float, EUCLIDEAN, rapid_matrix_mul<float> >::cpu_base_probe(int, int);

template<typename T,
        DistanceType dist_type,
        typename matrix_type>
cpu_base_probe<T, dist_type, matrix_type>::~cpu_base_probe(){
    delete this->mm;
}
template cpu_base_probe<int8_t, COSINE, base_matrix_mul<int8_t> >::~cpu_base_probe();
template cpu_base_probe<float, COSINE, base_matrix_mul<float> >::~cpu_base_probe();
template cpu_base_probe<int8_t, EUCLIDEAN, base_matrix_mul<int8_t>>::~cpu_base_probe();
template cpu_base_probe<float, EUCLIDEAN, base_matrix_mul<float> >::~cpu_base_probe();

template cpu_base_probe<int8_t, COSINE, rapid_matrix_mul<int8_t> >::~cpu_base_probe();
template cpu_base_probe<float, COSINE, rapid_matrix_mul<float> >::~cpu_base_probe();
template cpu_base_probe<int8_t, EUCLIDEAN, rapid_matrix_mul<int8_t> >::~cpu_base_probe();
template cpu_base_probe<float, EUCLIDEAN, rapid_matrix_mul<float> >::~cpu_base_probe();

template<typename T,
        DistanceType dist_type,
        typename matrix_type>
int cpu_base_probe<T, dist_type, matrix_type>::create_gallery(gallery<T> ** ga_ptr){
    cpu_base_gallery<T, dist_type> * ga = new cpu_base_gallery<T, dist_type>(this->dimension);
    //int ret = ga->init();
    (*ga_ptr) = (gallery<T>*)ga;
    return 0;
}
template int cpu_base_probe<int8_t, COSINE, base_matrix_mul<int8_t> >::create_gallery(gallery<int8_t> ** ga_ptr);
template int cpu_base_probe<float, COSINE, base_matrix_mul<float> >::create_gallery(gallery<float> ** ga_ptr);
template int cpu_base_probe<int8_t, EUCLIDEAN, base_matrix_mul<int8_t>>::create_gallery(gallery<int8_t> ** ga_ptr);
template int cpu_base_probe<float, EUCLIDEAN, base_matrix_mul<float> >::create_gallery(gallery<float> ** ga_ptr);

template int cpu_base_probe<int8_t, COSINE, rapid_matrix_mul<int8_t> >::create_gallery(gallery<int8_t> ** ga_ptr);
template int cpu_base_probe<float, COSINE, rapid_matrix_mul<float> >::create_gallery(gallery<float> ** ga_ptr);
template int cpu_base_probe<int8_t, EUCLIDEAN, rapid_matrix_mul<int8_t> >::create_gallery(gallery<int8_t> ** ga_ptr);
template int cpu_base_probe<float, EUCLIDEAN, rapid_matrix_mul<float> >::create_gallery(gallery<float> ** ga_ptr);

template<typename T,
        DistanceType dist_type,
        typename matrix_type>
int cpu_base_probe<T, dist_type, matrix_type>::query(const T * const x, const int n, gallery<T> * ga, Tout *sims, idx_t *idx){
    cpu_base_gallery<T, dist_type>* c_ga = (cpu_base_gallery<T, dist_type>*) ga;
    int num = c_ga->num;
    pair<Tout, idx_t>* res;
    vector<vector<pair<Tout, idx_t> > >ans(this->max_batch);
    T* data= (T*)c_ga->data.data();
    Tout* offset = (Tout*)c_ga->offset.data();

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
    for (int i = 0 ; i < n ; i += this->max_batch){
        int pn = std::min(this->max_batch, n - i);

        memcpy(this->x_tmp.data(), x + 1LL * i * this->dimension, pn * this->dimension * sizeof(T));
        if (is_same_type<T, int8_t>() == true){
            for (int64_t k = 0; k < 1LL * pn * this->dimension; ++k){
                this->x_tmp[k] += 64;
            }
        }

        for (int j = 0; j < num ; j += this->max_block){
            int block_size = std::min(this->max_block, num - j);
            this->mm->mul(this->x_tmp.data(), data + 1LL * j * this->dimension, offset + j, 
                        std::min(this->max_batch, n - i),  block_size, &res);
            
            for (int k = 0; k < std::min(this->max_batch, n - i); ++k)
                for (int l = 0; l < this->topk; ++l)
                    ans[k].push_back(std::make_pair(res[k * this->topk + l].first, res[k * this->topk + l].second + j));
        }

        for (int k = 0; k < pn; ++k){
            //std::cout << c_ga->offset[ans[k][0].second] << " ";
            std::nth_element(ans[k].data(), ans[k].data() + this->topk, ans[k].data() + ans[k].size(),
                             pair_greator<Tout, idx_t>());
            std::sort(ans[k].data(), ans[k].data() + this->topk, pair_greator<Tout, idx_t>());
            for (int j = 0 ; j < this->topk ; ++j){
                sims[(i + k) * this->topk + j] = ans[k][j].first;
                //sims[(i + k) * this->topk + j] = vec_dis<T, dist_type>(x + 1LL * i * this->dimension, data + 1LL * c_ga->ids[ans[k][j].second] * dimension, dimension);
                idx[(i + k) * this->topk + j] = c_ga->ids[ans[k][j].second];
            }
            ans[k].clear();
        }
    }
    //std::cout << std::endl;
    return 0;
}
template int cpu_base_probe<int8_t, COSINE, base_matrix_mul<int8_t> >::query(const int8_t * const x, const int n, gallery<int8_t> * ga, int *sims, idx_t *idx);
template int cpu_base_probe<float, COSINE, base_matrix_mul<float> >::query(const float * const x, const int n, gallery<float> * ga, float *sims, idx_t *idx);
template int cpu_base_probe<int8_t, EUCLIDEAN, base_matrix_mul<int8_t>>::query(const int8_t * const x, const int n, gallery<int8_t> * ga, int *sims, idx_t *idx);
template int cpu_base_probe<float, EUCLIDEAN, base_matrix_mul<float> >::query(const float * const x, const int n, gallery<float> * ga, float *sims, idx_t *idx);

template int cpu_base_probe<int8_t, COSINE, rapid_matrix_mul<int8_t> >::query(const int8_t * const x, const int n, gallery<int8_t> * ga, int *sims, idx_t *idx);
template int cpu_base_probe<float, COSINE, rapid_matrix_mul<float> >::query(const float * const x, const int n, gallery<float> * ga, float *sims, idx_t *idx);
template int cpu_base_probe<int8_t, EUCLIDEAN, rapid_matrix_mul<int8_t>>::query(const int8_t * const x, const int n, gallery<int8_t> * ga, int *sims, idx_t *idx);
template int cpu_base_probe<float, EUCLIDEAN, rapid_matrix_mul<float> >::query(const float * const x, const int n, gallery<float> * ga, float *sims, idx_t *idx);

template<typename T,
        DistanceType dist_type,
        typename matrix_type>
int cpu_base_probe<T, dist_type, matrix_type>::query_with_uids(const T* const x, const int n, gallery<T> * ga, idx_t *uids, const int m, Tout *sims, idx_t *idx){
    
    /* TODO */
    
    return NO_SUPPORT;
}
template int cpu_base_probe<int8_t, COSINE, base_matrix_mul<int8_t> >::query_with_uids(const int8_t * const x, const int n, gallery<int8_t> * ga, idx_t *uids, const int m, int *sims, idx_t *idx);
template int cpu_base_probe<float, COSINE, base_matrix_mul<float> >::query_with_uids(const float * const x, const int n, gallery<float> * ga, idx_t *uids, const int m, float *sims, idx_t *idx);
template int cpu_base_probe<int8_t, EUCLIDEAN, base_matrix_mul<int8_t>>::query_with_uids(const int8_t * const x, const int n, gallery<int8_t> * ga, idx_t *uids, const int m, int *sims, idx_t *idx);
template int cpu_base_probe<float, EUCLIDEAN, base_matrix_mul<float> >::query_with_uids(const float * const x, const int n, gallery<float> * ga, idx_t *uids, const int m, float *sims, idx_t *idx);

template int cpu_base_probe<int8_t, COSINE, rapid_matrix_mul<int8_t> >::query_with_uids(const int8_t * const x, const int n, gallery<int8_t> * ga, idx_t *uids, const int m, int *sims, idx_t *idx);
template int cpu_base_probe<float, COSINE, rapid_matrix_mul<float> >::query_with_uids(const float * const x, const int n, gallery<float> * ga, idx_t *uids, const int m, float *sims, idx_t *idx);
template int cpu_base_probe<int8_t, EUCLIDEAN, rapid_matrix_mul<int8_t>>::query_with_uids(const int8_t * const x, const int n, gallery<int8_t> * ga, idx_t *uids, const int m, int *sims, idx_t *idx);
template int cpu_base_probe<float, EUCLIDEAN, rapid_matrix_mul<float> >::query_with_uids(const float * const x, const int n, gallery<float> * ga, idx_t *uids, const int m, float *sims, idx_t *idx);
}
