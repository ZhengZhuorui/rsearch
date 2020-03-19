#pragma once
#include "rsearch_def.h"
#include "utils/avx2_asm.h"
namespace rsearch{

template<typename T,
        DistanceType dist_type,
        typename matrix_type>
class cpu_base_probe : public base_probe<T, dist_type, matrix_type>{
public:
    cpu_base_probe(int dimension, int topk);
    ~cpu_base_probe();
    using Tout = typemap_t<T>;
    virtual int create_gallery(gallery<T, dist_type> ** ga_ptr) override;
    virtual int query(const T * const x, const int n, gallery<T, dist_type> * ga, Tout *sims, uint32_t *idx) override;
    virtual int query_with_uids(const T* const x, const int n, gallery<T, dist_type> * ga, uint32_t *uids, const int m, Tout *sims, uint32_t *idx);
private:
    matrix_mul<T>* mm;
    int32_t max_batch, max_block, topk, dimension;
};


template<typename T,
        DistanceType dist_type,
        typename matrix_type>
cpu_base_probe<T, dist_type, matrix_type>::cpu_base_probe(int dimension, int topk) : base_probe<T, dist_type, matrix_type>(){
    this->mm = new matrix_type;
    this->dimension = dimension;
    this->topk = topk;
    this->max_batch = 32;
    this->max_block = 102400;
    this->mm->set(this->dimension, this->topk, this->max_batch, this->max_block);
}

template<typename T,
        DistanceType dist_type,
        typename matrix_type>
cpu_base_probe<T, dist_type, matrix_type>::~cpu_base_probe(){
    delete this->mm;
}

template<typename T,
        DistanceType dist_type,
        typename matrix_type>
int cpu_base_probe<T, dist_type, matrix_type>::create_gallery(gallery<T, dist_type> ** ga_ptr){
    cpu_base_gallery<T, dist_type> * ga = new cpu_base_gallery<T, dist_type>(this->dimension);
    //int ret = ga->init();
    (*ga_ptr) = (gallery<T, dist_type>*)ga;
    return 0;
}


template<typename T,
        DistanceType dist_type,
        typename matrix_type>
int cpu_base_probe<T, dist_type, matrix_type>::query(const T * const x, const int n, gallery<T, dist_type> * ga, Tout *sims, uint32_t *idx){
    cpu_base_gallery<T, dist_type>* c_ga = (cpu_base_gallery<T, dist_type>*) ga;
    int num = c_ga->num;
    pair<Tout, idx_t>* res;
    vector<vector<pair<Tout, idx_t> > >ans(this->max_batch);
    T* data= (T*)c_ga->data.data();
    Tout* offset = (Tout*)c_ga->offset.data();
    if (num < this->topk){
        ans.reserve(n);
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
        for (int j = 0; j < num ; j += this->max_block){
            int block_size = std::min(this->max_block, num - j);
            this->mm->mul(&x[i * this->dimension], &data[j * this->dimension], &offset[j], 
                        std::min(this->max_batch, n - i),  block_size, &res);
            for (int k = 0; k < std::min(this->max_batch, n - i); ++k)
                for (int l = 0; l < this->topk; ++l)
                    ans[k].push_back(std::make_pair(res[k * this->max_block + l].first, res[k * this->max_block + l].second + j));
        }

        for (int k = 0; k < std::min(this->max_batch, n - i); ++k){
            std::nth_element(ans[k].data(), ans[k].data() + this->topk + 1, ans[k].data() + ans[k].size(),
                             pair_greator<Tout, idx_t>());
            if (dist_type == COSINE)
                std::sort(ans[k].begin(), ans[k].end(), pair_greator<Tout, idx_t>());
            else 
                std::sort(ans[k].begin(), ans[k].end());
            for (int j = 0 ; j < this->topk ; ++j){
                sims[(i + k) * this->topk + j] = ans[k][j].first;
                idx[(i + k) * this->topk + j] = c_ga->ids[ans[k][j].second];
            }
            ans[k].clear();
        }
    }
    return 0;
}
template<typename T,
        DistanceType dist_type,
        typename matrix_type>
int cpu_base_probe<T, dist_type, matrix_type>::query_with_uids(const T* const x, const int n, gallery<T, dist_type> * ga, uint32_t *uids, const int m, Tout *sims, uint32_t *idx){
    
    /* TODO */
    
    return NO_SUPPORT;
}

}