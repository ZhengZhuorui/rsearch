#pragma once
#include "probe/rsearch_probe.h"
#include "probe/cpu_base_probe.h"
#include <bits/stdc++.h>

namespace rsearch{
using std::vector;
using std::unordered_map;
template<typename T,
        DistanceType dist_type>
class cpu_base_gallery : public gallery<T, dist_type>{
public:
    using Tout = typemap_t<T>;
    cpu_base_gallery(int dimension);

    virtual ~cpu_base_gallery();

    virtual int init() override;

    virtual int add(const T* const x, const int n) override;

    virtual int add_with_uids(const T* const x, const idx_t * const uids, const int n) override;

    virtual int change_by_uids(const T* const x, const idx_t * const uids, const int n) override;

    virtual int remove_by_uids(const idx_t * const uids, const int n) override;

    virtual int query_by_uids(const idx_t* const uids, const int n, T * x) override;

    virtual int reset() override;

private:
    vector<T> data;
    unordered_map<idx_t, idx_t> index;
    vector<idx_t> ids;
    vector<T> offset;

    uint32_t dimension;
    uint32_t num;
    idx_t max_id;
    
    std::mutex mtx;
    friend cpu_base_probe<T, dist_type>;
};

template<typename T,
        DistanceType dist_type>
cpu_base_gallery<T, dist_type>::cpu_base_gallery(int dimension) : gallery<T, dist_type>(){
    this->dimension = dimension;
    this->num = 0;
    this->max_id = 0;
}


template<typename T,
        DistanceType dist_type>
cpu_base_gallery<T, dist_type>::~cpu_base_gallery(){
    this->data.clear();
    this->index.clear();
    this->ids.clear();
    this->offset.clear();
}

template<typename T,
        DistanceType dist_type>
int cpu_base_gallery<T, dist_type>::init(){
    this->data.clear();
    this->index.clear();
    this->ids.clear();
    this->offset.clear();
    this->num = 0;
    return 0;
}

template<typename T,
        DistanceType dist_type>
int cpu_base_gallery<T, dist_type>::add(const T* const x, const int n){
    this->mtx.lock();
    this->data.reserve((this->num + n) * this->dimension);
    memcpy(this->data.data() + 1LL * this->num * this->dimension, x, 1LL * sizeof(T) * n * this->dimension);
    this->offset.reserve((this->num+n));
    for (int i = 0; i < n; ++i){
        this->index[this->max_id] = this->num + i;
        this->ids.push_back(this->max_id);
        this->max_id++;
    }
    for (int i = 0 ; i < n ; ++i){
        this->offset[this->num + i] = get_offset<T, dist_type>(this->data.data() + 1LL * (this->num + i) * this->dimension, this->data.data() + 1LL * (this->num + i) * this->dimension, this->dimension);
    }    
    this->num += n;
    this->mtx.unlock();
    return 0;
}

template<typename T,
        DistanceType dist_type>
int cpu_base_gallery<T, dist_type>::add_with_uids(const T* const x, const idx_t * const uids, const int n){
    this->mtx.lock();
    for (int i = 0; i < n; ++i){
        if (this->index.find(uids[i]) != this->index.end()){
            return INDEX_EXISTS;
        }
    }
    for (int i = 0; i < n; ++i){
        this->ids.push_back(uids[i]);
        this->index[uids[i]] = num;
        this->max_id = std::max(this->max_id, uids[i] + 1);
    }
    
    this->data.reserve((this->num + n) * this->dimension);
    memcpy(this->data.data() + 1LL * this->num * this->dimension, x, 1LL * sizeof(T) * n * this->dimension);
    this->offset.reserve(this->num + n);
    for (int i = 0 ; i < n ; ++i){
        this->offset[this->num + i] = get_offset<T, dist_type>(this->data.data() + 1LL * (this->num + i) * this->dimension, this->data.data() + 1LL * (this->num + i) * this->dimension, this->dimension);
    }  
    this->num += n;
    this->mtx.unlock();
    return 0;
}

template <typename T,
        DistanceType dist_type>
int cpu_base_gallery<T, dist_type>::change_by_uids(const T* const x, const idx_t * const uids, const int n){
    this->mtx.lock();
    for (int i = 0; i < n; ++i){
        if (this->index.find(uids[i]) == this->index.end())
            return INDEX_NO_FIND;
    }
    for (int i = 0; i < n; ++i){
        memcpy(this->data.data() + 1LL * this->index[uids[i]] * this->dimension, x + 1LL * i * dimension, sizeof(T) * this->dimension);
        this->offset[this->index[uids[i]]] = get_offset<T>(this->data.data() + 1LL * this->index[uids[i]] * this->dimension, this->data.data() + 1LL * this->index[uids[i]] * this->dimension, this->dimension);
    }
    this->mtx.unlock();
    return 0;
}

template<typename T,
        DistanceType dist_type>
int cpu_base_gallery<T, dist_type>::remove_by_uids(const idx_t* const uids, const int n){
    this->mtx.lock();
    for (int i = 0; i < n; ++i){
        if (this->index.find(uids[i]) == this->index.end())
            return INDEX_NO_FIND;
    }
    for (int i = 0; i < n ; ++i){
        int p = this->index[uids[i]];
        memcpy(this->data.data() + 1LL * p * this->dimension, this->data.data() + 1LL * (this->num - 1) * this->dimension, sizeof(T) * this->dimension);
        this->offset[p] = this->offset[num - 1];
        this->index[this->ids[this->num - 1]] = p;
        this->ids[p] = this->ids[this->num - 1];
        this->index.erase(uids[i]);
        this->ids.pop_back();
        this->num--;
    }
    this->data.reserve(this->num * this->dimension);
    this->offset.reserve(this->num);
    this->mtx.unlock();
    return 0;   
}

template<typename T,
        DistanceType dist_type>
int cpu_base_gallery<T, dist_type>::query_by_uids(const idx_t* const uids, const int n, T * x){
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
int cpu_base_gallery<T, dist_type>::reset(){
    this->data.clear();
    this->index.clear();
    this->ids.clear();
    this->offset.clear();
    this->num = 0;
    return 0;
}



}