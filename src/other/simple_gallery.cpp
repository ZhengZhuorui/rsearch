#include "other/other.h"
#include "other/simple_gallery.h"
namespace rsearch{
template<typename T>
simple_gallery<T>::simple_gallery() : gallery<T>(){
    this->num = 0;
    this->max_id = 0;
}

template<typename T>
simple_gallery<T>::~simple_gallery(){
    this->data.clear();
    this->index.clear();
    this->ids.clear();
}

template<typename T>
int simple_gallery<T>::init(){
    this->data.clear();
    this->index.clear();
    this->ids.clear();
    this->num = 0;
    return 0;
}

template<typename T>
int simple_gallery<T>::add(const T* const x, const int n){
    this->mtx.lock();
    this->data.resize(1LL * (this->num + n));
    memcpy(this->data.data() + 1LL * this->num, x, 1LL * sizeof(T) * n);
    for (int i = 0; i < n; ++i){
        this->index[this->max_id] = this->num + i;
        this->ids.push_back(this->max_id);
        this->max_id++;
    }
    this->num += n;
    this->mtx.unlock();
    return 0;
}


template<typename T>
int simple_gallery<T>::add_with_uids(const T* const x, const idx_t * const uids, const int n){
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
    
    this->data.resize(1LL * (this->num + n));
    memcpy(this->data.data() + 1LL * this->num, x, 1LL * sizeof(T) * n);
    this->num += n;
    this->mtx.unlock();
    return 0;
}

template <typename T>
int simple_gallery<T>::change_by_uids(const T* const x, const idx_t * const uids, const int n){
    this->mtx.lock();
    for (int i = 0; i < n; ++i){
        if (this->index.find(uids[i]) == this->index.end())
            return INDEX_NO_FIND;
    }
    for (int i = 0; i < n; ++i){
        this->data[this->index[uids[i]]] = *x;
    }
    this->mtx.unlock();
    return 0;
}

template<typename T>
int simple_gallery<T>::remove_by_uids(const idx_t* const uids, const int n){
    this->mtx.lock();
    for (int i = 0; i < n; ++i){
        if (this->index.find(uids[i]) == this->index.end())
            return INDEX_NO_FIND;
    }
    for (int i = 0; i < n ; ++i){
        int p = this->index[uids[i]];
        this->data[p] = this->data[this->num - 1];
        this->index[this->ids[this->num - 1]] = p;
        this->ids[p] = this->ids[this->num - 1];
        this->index.erase(uids[i]);
        this->ids.pop_back();
        this->num--;
    }
    this->data.resize(1LL * this->num);
    this->mtx.unlock();
    return 0;   
}

template<typename T>
int simple_gallery<T>::query_by_uids(const idx_t* const uids, const int n, T * x){
    for (int i = 0; i < n; ++i){
        if (this->index.find(uids[i]) == this->index.end())
            return INDEX_NO_FIND;
    }
    for (int i = 0; i < n ; ++i){
        //memcpy(x, &this->data[this->index[uids[i]]], sizeof(T));
        x[i] = this->data[this->index[uids[i]]];
    }
    return 0;
}

template<typename T>
int simple_gallery<T>::reset(){
    this->data.clear();
    this->index.clear();
    this->ids.clear();
    this->num = 0;
    return 0;
}

template<typename T>
int simple_gallery<T>::load_data(std::string file_name){
    this->mtx.lock();
    ifstream fin(file_name, ifstream::binary);
    int type, d, n;
    r_read(fin, &type, 1);
    r_read(fin, &d, 1);
    if (type != SIMPLE_GALLERY)
        return LOAD_DATA_ERROR;
    r_read(fin, &n, 1);
    //std::cout << "[load data]" << n << std::endl; 
    vector<idx_t> ids_tmp(n);
    r_read(fin, ids_tmp.data(), n);
    for (int i = 0; i < n; ++i){
        if (this->index.find(ids_tmp[i]) != this->index.end())
            return INDEX_EXISTS;
    }
    this->ids.resize(this->num + n);
    memcpy(this->ids.data() + this->num, ids_tmp.data(), n * sizeof(idx_t));
    for (int i = 0; i < n; ++i){
        this->index[ids_tmp[i]] = this->num + i;
    }
    this->data.resize(1LL * (this->num + n));
    r_read(fin, this->data.data() + 1LL * this->num, 1LL * n);
    
    this->num += n;
    this->mtx.unlock();
    return 0;
}


template<typename T>
int simple_gallery<T>::store_data(std::string file_name){
    this->mtx.lock();
    ofstream fout(file_name, ofstream::binary);
    int type = SIMPLE_GALLERY;
    r_write(fout, &type, 1);
    r_write(fout, &this->num, 1);
    r_write(fout, this->ids.data(), this->num);
    r_write(fout, this->data.data(), 1LL * this->num);
    //std::cout << "[store data] " << this->num << " " << this->data[0] << " " << this->offset[0] << std::endl;
    this->mtx.unlock();
    return 0;
}

template simple_gallery<area_time>::simple_gallery();
template simple_gallery<area_time>::~simple_gallery();
template int simple_gallery<area_time>::init();
template int simple_gallery<area_time>::add(const area_time* const x, const int n);
template int simple_gallery<area_time>::add_with_uids(const area_time* const x, const idx_t * const uids, const int n);
template int simple_gallery<area_time>::change_by_uids(const area_time* const x, const idx_t * const uids, const int n);
template int simple_gallery<area_time>::remove_by_uids(const idx_t * const uids, const int n);
template int simple_gallery<area_time>::query_by_uids(const idx_t * const uids, const int n, area_time* x);
template int simple_gallery<area_time>::reset();
template int simple_gallery<area_time>::load_data(std::string file_name);
template int simple_gallery<area_time>::store_data(std::string file_name);

}
