#pragma once
#include "rsearch_def.h"
#include "utils/cluster.h"
#include <bits/stdc++.h>

namespace rsearch{
using std::vector;
using std::unordered_map;
struct pqivf_traits{
    int cq_num;
    int select_cq;
    int pq_dimension;
    int pq_num;
    float k;
};

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
    
    virtual int write_train_data(ofstream& fout);

    virtual int read_train_data(ifstream& fin);

private:

    int add_one(const pair<Tout, idx_t>* const x, const int id, const int cq_id);

    vector<T> cq;
    vector<T> pq;
    vector<Tout> cq_offset;
    vector<Tout> pq_offset;
    vector<float> cq_float;
    vector<float> pq_float;

    vector<vector<uint8_t> > data;
    vector<int> block_num;
    unordered_map<idx_t, pair<int, int> > index;
    vector<vector<idx_t> > ids;

    int dimension;
    int num;
    idx_t max_id;
    bool have_train_;

    int cq_num;
    int select_cq;
    int pq_dimension;
    int pq_num;
    int code_len;
    float k;
    int max_batch, max_block;
    //int8_t* ix;
    std::mutex mtx;

    rapid_matrix_mul<T>* cq_mm;
    rapid_matrix_mul<T>* pq_mm;

    friend pqivf_probe<T, dist_type>;
};

template<typename T,
        DistanceType dist_type>
pqivf_gallery<T, dist_type>::pqivf_gallery(int dimension, struct pqivf_traits& traits) : gallery<T, dist_type>(){
    this->dimension = dimension;
    this->num = 0;
    this->max_id = 0;
    this->k = traits.k;
    this->cq_num = traits.cq_num;
    this->select_cq = traits.select_cq;
    this->pq_num = traits.pq_num;
    this->pq_dimension = traits.pq_dimension;
    this->code_len = this->dimension / this->pq_dimension;
    this->cq.reserve(this->cq_num);
    this->pq.reserve(this->pq_num * this->dimension);
    this->cq_offset.reserve(this->cq_num);
    this->pq_offset.reserve(this->pq_num);
    this->block_num.reserve(this->cq_num);
    this->cq_mm = new rapid_matrix_mul<T>();
    this->pq_mm = new rapid_matrix_mul<T>();
    this->max_batch = 32;
    this->max_block = 512000;
    //this->ix = (int8_t*)malloc(this->max_batch * this->code_len);

    this->cq_mm->set(this->dimension, 1, this->max_batch, this->cq_num);
    this->pq_mm->set(this->pq_dimension, 1, this->max_batch * this->code_len, this->pq_num);
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
        return NO_TRAIN;
    }
    if (is_same_type<T,int8_t>() == true){
        float_7bits(this->cq_float.data(), (int8_t*)this->cq.data(), this->cq_num * this->dimension);
        float_7bits(this->pq_float.data(), (int8_t*)this->pq.data(), this->pq_num * this->pq_dimension);
    }
    else{
        memcpy(this->cq.data(), this->cq_float.data(), this->cq_num * this->dimension * sizeof(T) );
        memcpy(this->pq.data(), this->cq_float.data(), this->cq_num * this->dimension * sizeof(T) );
    }
    for (int i = 0; i < cq_num; ++i){
            this->cq_offset[i] = get_offset<T, dist_type>(this->cq.data() + 1LL * i * this->dimension, this->dimension);
        }
        for (int i = 0; i < pq_num; ++i){
            this->pq_offset[i] = get_offset<T, dist_type>(this->pq.data() + 1LL * i * this->pq_dimension, this->pq_dimension);
        }
    return 0;
}

template<typename T,
        DistanceType dist_type>
int pqivf_gallery<T, dist_type>::reset(){
    this->data.clear();
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
    for (int i = 0; i < n; i += this->max_batch){
        pair<Tout, idx_t>* cq_res;
        pair<Tout, idx_t>* pq_res;
        int qn = std::min(this->max_batch, n - i);
        this->cq_mm->mul(x + 1LL * i * this->dimension, this->cq.data(), this->cq_offset.data(), qn, this->cq_num, &cq_res);
        //get_code_v1<T,int8_t>(x + 1LL * i * this->dimension, this->pq.data(), this->pq_offset.data() n, this->pq_num, this->pq_dimension, this->dimension, this->ix);
        this->pq_mm->mul(x + 1LL * i * this->dimension, this->pq.data(), this->pq_offset.data(), qn * this->code_len, 
                        this->pq_num, &pq_res);
        for (int j = 0; j < qn; ++j)
            this->add_one(pq_res + 1LL * this->code_len * this->pq_num, this->max_id++, cq_res[j * this->cq_num].second);
    }
    this->mtx.unlock();
    return 0;
}

template<typename T,
        DistanceType dist_type>
int pqivf_gallery<T, dist_type>::add_one(const pair<Tout, idx_t>* const x, const int id, const int cq_id){
    this->data[cq_id].reserve((this->block_num[cq_id] + 1) * this->dimension);
    //memcpy(this->data[cq_id].data() + 1LL * this->block_num[cq_id] * this->code_len, x, this->code_len * sizeof(uint8_t));
    for (int j = 0; j < this->code_len; ++j)
        this->data[cq_id][1LL * this->block_num[cq_id] * this->code_len + j] = x[j * this->pq_num].second;
    this->ids[cq_id].push_back(id);
    this->index[id] = std::make_pair(cq_id, this->block_num[cq_id]);
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
    for (int i = 0; i < n; i += this->max_batch){
        pair<Tout, idx_t>* cq_res;
        pair<Tout, idx_t>* pq_res;
        int qn = std::min(this->max_batch, n - i);

        this->cq_mm->mul(x + 1LL * i * this->dimension, this->cq.data(), this->cq_offset.data(), this->max_batch, this->cq_num, &cq_res);
        this->pq_mm->mul(x + 1LL * i * this->dimension, this->pq.data(), this->pq_offset.data(), qn * this->code_len, 
                        this->pq_num, &pq_res);
            
        for (int j = 0; j < qn; ++j){
            this->add_one(pq_res + 1LL * this->code_len * this->pq_num, uids[i + j], cq_res[j * this->cq_num].second);
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
    this->remove_by_uids(uids, n);
    this->add_with_uids(x, uids, n);
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
        pair<int, int> p = this->index[uids[i]];
        memcpy(this->data[p.first].data() + 1LL * p.second * this->code_len, 
               this->data[p.first].data() + 1LL * (this->block_num[p.first] - 1) * this->code_len, this->code_len * sizeof(T));
        this->ids[p.first][p.second] = this->ids[p.first][this->block_num[p.first] - 1];
        this->index[this->ids[p.first][p.second]] = p;
        this->index.erase(uids[i]);
        this->block_num[p.first]--;
        this->data[p.first].reserve(this->block_num[p.first] * this->code_len);
    }
    this->num -= n;
    this->mtx.unlock();
    return 0;
}

template<typename T,
        DistanceType dist_type>
int pqivf_gallery<T, dist_type>::query_by_uids(const idx_t* const uids, const int n, T * x){
    /*
    for (int i = 0; i < n; ++i){
        if (this->index.find(uids[i]) == this->index.end())
            return INDEX_NO_FIND;
    }
    for (int i = 0; i < n ; ++i){
        pair<int, int> p = this->index[uids[i]];
        memcpy(x, &this->data[p.first][p.second], sizeof(T) * this->dimension);
    }*/
    return NO_SUPPORT;
}

template<typename T,
        DistanceType dist_type>
int pqivf_gallery<T, dist_type>::train(const float* data, const int n, const int dimension){
    k_means<float, dist_type>(data, n, this->cq_num, this->dimension, this->cq_float);
    k_means<float, dist_type>(data, n * this->code_len, this->pq_num, this->pq_dimension, this->pq_float);
    this->have_train_ = true;
    this->init();
    return 0;
}

template<typename T,
        DistanceType dist_type>
int pqivf_gallery<T, dist_type>::write_train_data(ofstream& fout){
    int dt = dist_type;
    if (this->have_train_ == false)
        return NO_TRAIN;
    r_write(fout, &this->dimension, 1);

    r_write(fout, &dt, 1);
    struct pqivf_traits traits = {this->cq_num, this->select_cq, this->pq_dimension, this->pq_num, this->k};
    r_write(fout, &traits, 1);
    r_write(fout, this->cq_float.data(), this->cq_num * this->dimension);
    r_write(fout, this->pq_float.data(), this->pq_num * this->pq_dimension);
    return 0;
}


template<typename T,
        DistanceType dist_type>
int pqivf_gallery<T, dist_type>::read_train_data(ifstream& fin){
    struct pqivf_traits traits;
    int d, dt;
    r_read(fin, &d, 1);
    r_read(fin, &dt, 1);
    r_read(fin, &traits, 1);
    if (d != this->dimension || dt != dist_type || traits.cq_num != this->cq_num || traits.select_cq != this->select_cq || 
        traits.pq_dimension != this->pq_dimension || traits.pq_num != this->pq_dimension)
    return TRAINDATA_ERROR;
    this->cq_float.reserve(this->cq_num * this->dimension);
    this->pq_float.reserve(this->pq_num * this->pq_dimension);
    r_read(fin, this->cq_float.data(), this->cq_num * this->dimension);
    r_read(fin, this->pq_float.data(), this->pq_num * this->pq_dimension);
    this->have_train_ = true;
    this->init();
    return 0;
}

}