#pragma once
#include "probe/rsearch_probe.h"
#include "gallery/rsearch_gallery.h"
#include "matrix/rapid_matrix_mul.h"
#include "probe/pqivf_probe.h"
#include <bits/stdc++.h>

namespace rsearch{
using std::vector;
using std::map;
struct pqivf_traits{
    int cq_num;
    int select_cq;
    int pq_dimension;
    int pq_num;
    float k;
};

template<typename T, DistanceType dist_type> class pqivf_probe;

template<typename T,
        DistanceType dist_type>
class pqivf_gallery : public gallery<T, dist_type>{
public:
    using Tout = typemap_t<T>;

    pqivf_gallery(int dimension, struct pqivf_traits& traits);

    virtual ~pqivf_gallery();

    virtual int init() override;

    virtual int add(const T* const x, const int n) override;

    virtual int add_with_uids(const T* const x, const idx_t * const uids, const int n) override;

    virtual int change_by_uids(const T* const x, const idx_t * const uids, const int n) override;

    virtual int remove_by_uids(const idx_t * const uids, const int n) override;

    virtual int reset() override;

    virtual int query_by_uids(const idx_t* const uid, const int n, T * x) override;

    virtual int train(const float* const x, int n, int dimension);

    virtual bool have_train(){return this->have_train_;}
    
    virtual int write_train_data(ofstream& fout) override;

    virtual int read_train_data(ifstream& fin) override;

private:

    int add_one(const T* const x, const int id, const int cq_id);

    vector<T> cq;
    vector<T> pq;
    vector<Tout> cq_offset;
    vector<Tout> pq_offset;
    vector<float> cq_float;
    vector<float> pq_float;

    vector<vector<uint8_t> > data;
    vector<Tout> block_num;
    map<pair<int, int>, idx_t> index;
    vector<vector<idx_t> > ids;

    uint32_t dimension;
    uint32_t num;
    idx_t max_id;
    bool have_train_;
    int topk;

    int cq_num;
    int select_cq;
    int pq_dimension;
    int pq_num;
    int code_len;
    float k;
    int8_t* ix;
    std::mutex mtx;

    rapid_matrix_mul<int8_t>* cq_mm;

    friend pqivf_probe<T, dist_type>;
};

template<typename T,
        DistanceType dist_type>
pqivf_gallery<T, dist_type>::pqivf_gallery(int dimension, struct pqivf_traits& traits) : gallery<T, dist_type>(){
    this->dimension = dimension;
    this->num = 0;
    this->max_id = 0;
    this->dist_type = dist_type;
    this->k = traits.k;
    this->cq_num = traits.cq_num;
    this->select_cq = traits.select_cq;
    this->pq_num = traits.pq_num;
    this->pq_dimension = traits.pq_dimension;
    this->code_len = this->dimension / this->pq_dimension;
    this->cq.reserve(this->cq_num);
    this->pq.reserve(this->pq_num * this->dimension);
    this->pq_data.resize(this->cq_num);
    this->block_num.reserve(this->cq_num);
    this->cq_mm = new rapid_matrix_mul<int8_t>();
    this->max_batch = 32;
    this->max_block = 512000;
    this->ix = (int8_t*)malloc(this->max_batch * this->pq_dimension);
}

template<typename T,
        DistanceType dist_type>
pqivf_gallery<T, dist_type>::~pqivf_gallery(){
    delete this->cq_mm;
}

template<typename T,
        DistanceType dist_type>
int pqivf_gallery<T, dist_type>::init(){
    if (this->have_train_ == false){
        return NOT_TRAIN;
    }
    this->cq.reserve(this->cq_num * this->dimension);
    float_7bits(this->cq_float.data(), this->cq.data(), this->cq_num * this->dimension);
    this->cq_offset.reserve(this->cq_num);
    for (int i = 0; i < cq_num; ++i){
        this->cq_offset[i] = get_offset<int8_t, dist_type>(this->cq.data() + 1LL * i * this->dimension, this->cq.data() + 1LL * i * this->dimension, this->dimension);
    }
    this->cq_mm->set(this->dimension, 1, this->max_batch, this->cq_num);

    this->pq.reserve(this->pq_num * this->dimension);
    float_7bits(this->pq_float.data(), this->pq.data(), this->pq_num * this->pq_dimension);
    this->pq_offset.reserve(this->pq_num);
    for (int i = 0; i < pq_num; ++i){
        this->pq_offset[i] = get_offset<int8_t, dist_type>(this->pq.data() + 1LL * i * this->pq_dimension, this->pq.data() + 1LL * i * this->pq_dimension, this->pq_dimension);
    }

    //this->pq_mm->set(this->pq_dimension, 1, this->max_batch, this->pq_num);
    return 0;
}

template<typename T,
        DistanceType dist_type>
int pqivf_gallery<T, dist_type>::reset(){
    this->data.clear();
    for (int i = 0; i < this->pq_num; ++i)
        this->pq_data[i].clear();
    this->index.clear();
    this->ids.clear();
    this->cq.clear();
    this->pq.clear();
    this->cq_offset.clear();
    this->pq_offset.clear();
    this->block_num.clear();
    this->num = 0;
    this->max_id = 0;
    this->have_train_ = false;
    return 0;
}

template<typename T,
        DistanceType dist_type>
int pqivf_gallery<T, dist_type>::add(const T* const x, const int n){
    this->mtx.lock();
    for (int i = 0; i < n; i += this->batch){
        pair<int, idx_t>* res;
        this->cq_mm->mul(x + 1LL * i * this->dimension, this->cq.data(), this->cq.offset(), this->batch, this->cq_num, &res);
        get_code_v1(x + 1LL * i * this->dimension, this->pq.data(), this->n);
        
        for (int j = 0; j < std::min(this->batch, n - i); ++j)
            this->add_one(x + 1LL * (i + j) * this->dimension, this->max_id++, res[j * this->cq_num].second);
    }
    this->mtx.unlock();
    return 0;
}

template<typename T,
        DistanceType dist_type>
int pqivf_gallery<T, dist_type>::add_one(const T* const x, const int id, const int cq_id){
    this->data[cq_id].reserve((this->block_num[cq_id] + 1) * this->dimension);
    memcpy(this->data[cq_id].data() + 1LL * this->block_num[cq_id] * this->dimension, x, this->code_len * sizeof(T));
    this->ids[cq_id].push_back(id);
    this->index[id].push_back(std::make_pair(cq_id, this->block_num[cq_id]));
    this->block_num[cq_id]++;
    this->num++;
    return 0;
}


template<typename T,
        DistanceType dist_type>
int pqivf_gallery<T, dist_type>::add_with_uids(const T* const x, const idx_t * const uids, const int n){
    this->mtx.lock();
    for (int i = 0; i < n; ++i){
        if (this->index.find(uids[i]) != this->index.end()){
            return INDEX_EXISTS;
        }
    }
    for (int i = 0; i < n; i += this->batch){
        pair<int, idx_t>* res;
        this->cq_mm->mul(x + 1LL * i * this->dimension, this->cq.data(), this->cq.offset(), this->batch, this->cq_num, &res);
        get_code_v1(x + 1LL * i * this->dimension, this->pq.data(), this->n);
        for (int j = 0; j < std::min(this->batch, n - i); ++j){
            this->add_one(x + 1LL * (i + j) * this->dimension, uids[i + j], res[j * this->cq_num].second);
            this->max_id = std::max(this->max_id, uids[i + j] + 1);
        }
    }
    this->mtx.unlock();
    return 0;
}

template <typename T,
        DistanceType dist_type>
int pqivf_gallery<T, dist_type>::change_by_uids(const T* const x, const idx_t * const uids, const int n){
    for (int i = 0; i < n; ++i){
        if (this->index.find(uids[i]) == this->index.end())
            return INDEX_NO_FIND;
    }
    for (int i = 0; i < n; ++i){
        this->remove_by_uids(uids, n);
        this->add_with_uids(x, uids, n);

    }
    return 0;
}


template<typename T,
        DistanceType dist_type>
int pqivf_gallery<T, dist_type>::remove_by_uids(const idx_t* const uids, const int n){
    this->mtx.lock();
    for (int i = 0; i < n; ++i){
        if (this->index.find(uids[i]) == this->index.end())
            return INDEX_NO_FIND;
    }
    for (int i = 0; i < n; ++i){
        pair<int, int> p = this->index.find(uids[i]);
        memcpy(this->data[p.first].data() + 1LL * p.second * this->code_len, 
               this->data[p.first].data() + 1LL * (this->block_num[p.first] - 1)* this->code_len, this->code_len * sizeof(T));
        this->ids[p.first][p.second] = this->ids[p.first][this->block_num[p.first] - 1];
        this->index[this->ids[p.first][p.second]] = p;
        this->index.erase(uids[i]);
    }
    this->num -= n;
    this->mtx.unlock();
    return 0;
}

template<typename T,
        DistanceType dist_type>
int pqivf_gallery<T, dist_type>::query_by_uids(const idx_t* const uids, const int n, T * x){
    for (int i = 0; i < n; ++i){
        if (this->index.find(uids[i]) == this->index.end())
            return INDEX_NO_FIND;
    }
    for (int i = 0; i < n ; ++i){
        memcpy(x, &this->data[this->index[uids[i]]], sizeof(T) * this->dimension);
    }
    return 0;
}

template<typename T,
        DistanceType dist_type>
int pqivf_gallery<T, dist_type>::train(const float* data, const int n, const int dimension){
    k_means(data, n, this->cq_num, this->dimension, this->cq_float);
    k_means(data, n * this->code_len, this->pq_dimension, this->pq_float);
    this->have_train_ = true;
    this->init();
    return 0;
}

template<typename T,
        DistanceType dist_type>
int pqivf_gallery<T, dist_type>::write_train_data(std::ofstream& fout){
    int dt = this->dist_type;
    if (this->have_train_ == false)
        return NOT_TRAIN;
    r_write(fout, &this->dimension, 1);

    r_write(fout, &dt, 1);
    struct pqivf_traits traits(this->cq_num, this->select_cq, this->pq_dimension, this->pq_num, this->k);
    r_write(fout, &traits, 1);
    r_write(fout, this->cq_y.data(), this->cq_num * this->dimension);
    r_write(fout, this->pq_y.data(), this->pq_num * this->pq_dimension);
    return 0;
}


template<typename T,
        DistanceType dist_type>
int pqivf_gallery<T, dist_type>::read_train_data(std::ifstream& fin){
    struct pqivf_traits traits;
    int d, dt;
    r_read(fin, &d, 1);
    r_read(fin, &dt, 1);
    r_read(fin, &traits, 1);
    if (d != this->dimension || dt != this->dist_type || traits.cq_num != this->cq_num || traits.select_cq != this->select_cq || 
        traits.pq_dimension != this->pq_dimension || traits.pq_num != this->pq_dimension)
    return TRAINDATA_ERROR;
    this->cq_y.reserve(this->cq_num * this->dimension);
    this->pq_y.reserve(this->pq_num * this->pq_dimension);
    r_read(fin, this->cq_y.data(), this->cq_num * this->dimension);
    r_read(fin, this->pq_y.data(), this->pq_num * this->pq_dimension);
    this->have_train_ = true;
    this->init();
    return 0;
}

}