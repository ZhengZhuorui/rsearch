#include "gallery/ivf_gallery.h"
#include "utils/helpers.h"
#include "utils/utils.h"
namespace rsearch{
using std::ofstream;
using std::ifstream;
using std::unordered_map;
using std::vector;
using std::pair;
template<typename T,
        DistanceType dist_type>
ivf_gallery<T, dist_type>::ivf_gallery(int dimension, struct ivf_traits& traits) : gallery<T>(){
    this->dimension = dimension;
    this->num = 0;
    this->max_id = 0;
    this->have_train_ = false;
    this->cq_num = traits.cq_num;
    this->select_cq = traits.select_cq;

    this->max_batch = 32;
    this->max_block = 512000;
    
    this->cq.resize(this->cq_num * this->dimension);
    this->cq_offset.resize(this->cq_num);
    this->cq_float.resize(this->cq_num * this->dimension);

    this->block_num.resize(this->cq_num);
    this->ids.resize(this->cq_num);
    this->data.resize(this->cq_num);
    this->x_tmp.resize(this->max_batch * this->dimension);
    this->index.clear();
    this->cq_mm = new rapid_matrix_mul<T>();
    this->cq_mm->set(this->dimension, 1, this->max_batch, this->cq_num);

}
template ivf_gallery<int8_t, COSINE>::ivf_gallery(int, struct ivf_traits&);
template ivf_gallery<float, COSINE>::ivf_gallery(int, struct ivf_traits&);
template ivf_gallery<int8_t, EUCLIDEAN>::ivf_gallery(int, struct ivf_traits&);
template ivf_gallery<float, EUCLIDEAN>::ivf_gallery(int, struct ivf_traits&);

template<typename T,
        DistanceType dist_type>
ivf_gallery<T, dist_type>::~ivf_gallery(){
    delete this->cq_mm;
    delete this->pq_mm;
}
template ivf_gallery<int8_t, COSINE>::~ivf_gallery();
template ivf_gallery<float, COSINE>::~ivf_gallery();
template ivf_gallery<int8_t, EUCLIDEAN>::~ivf_gallery();
template ivf_gallery<float, EUCLIDEAN>::~ivf_gallery();

template<typename T,
        DistanceType dist_type>
int ivf_gallery<T, dist_type>::init(){
    std::string dist_type_name = GetDistancetypeName<dist_type>();
    std::string train_file_name = "/home/zhengzhuorui/project/data/ivf_train_data." + dist_type_name + "." + std::to_string(this->cq_num) + "." + \
        std::to_string(this->pq_dimension) + "." + std::to_string(this->pq_num) + ".bin";
    if (this->have_train_ == false){  
        if (file_exist(train_file_name) == true){
            int code = this->load_train_data(train_file_name);
            if (code == TRAINDATA_ERROR)
                return TRAINDATA_ERROR;
        }
        else{
            vector<float> data;
            std::cout << "[init] target 2" <<std::endl;
            get_random_data<float, dist_type>(data, 200000, this->dimension);
            this->train(data.data(), 200000, this->dimension);
            this->store_train_data(train_file_name);            
        }
    }
    if (is_same_type<T, int8_t>() == true){
        float_7bits(this->cq_float.data(), (int8_t*)this->cq.data(), this->cq_num * this->dimension);
    }
    else{
        memcpy(this->cq.data(), this->cq_float.data(), this->cq_num * this->dimension * sizeof(T) );
    }
    std::cout << "[init] target 2" << std::endl;
    for (int i = 0; i < cq_num; ++i){
        this->cq_offset[i] = get_offset<T, dist_type>(this->cq.data() + 1LL * i * this->dimension, this->dimension);
    }
    return 0;
}
template int ivf_gallery<int8_t, COSINE>::init();
template int ivf_gallery<float, COSINE>::init();
template int ivf_gallery<int8_t, EUCLIDEAN>::init();
template int ivf_gallery<float, EUCLIDEAN>::init();

template<typename T,
        DistanceType dist_type>
int ivf_gallery<T, dist_type>::reset(){
    this->index.clear();
    this->cq.clear();
    this->cq_offset.clear();
    this->cq_float.clear();
    for (int i = 0; i < this->cq_num; ++i){
        this->block_num[i] = 0;
        this->data[i].clear();
        this->ids[i].clear();
    }
    
    this->num = 0;
    this->max_id = 0;
    this->have_train_ = false;
    return 0;
}
template int ivf_gallery<int8_t, COSINE>::reset();
template int ivf_gallery<float, COSINE>::reset();
template int ivf_gallery<int8_t, EUCLIDEAN>::reset();
template int ivf_gallery<float, EUCLIDEAN>::reset();

template<typename T,
        DistanceType dist_type>
int ivf_gallery<T, dist_type>::add(const T* const x, const int n){
    this->mtx.lock();
    pair<Tout, idx_t>* cq_res;
    pair<Tout, idx_t>* pq_res;
    for (int i = 0; i < n; i += this->max_batch){
        int qn = std::min(this->max_batch, n - i);
        memcpy(x_tmp.data(), x + 1LL * i * this->dimension, 1LL * qn * this->dimension * sizeof(T));
        if (is_same_type<T, int8_t>() == true){
            for (int k = 0; k < 1LL * qn * this->dimension; ++k)
                x_tmp[k] += 64;
        }
        this->cq_mm->mul(x_tmp.data(), this->cq.data(), this->cq_offset.data(), qn, this->cq_num, &cq_res);
        for (int j = 0; j < qn; ++j)
            this->add_one(this->max_id++, cq_res[j].second, x_tmp + 1LL * j * this->dimension);
    }
    //std::cout << "[add]" << cq_res[0].second << std::endl;
    this->mtx.unlock();
    return 0;
}
template int ivf_gallery<int8_t, COSINE>::add(const int8_t* const x, const int n);
template int ivf_gallery<float, COSINE>::add(const float* const x, const int n);
template int ivf_gallery<int8_t, EUCLIDEAN>::add(const int8_t* const x, const int n);
template int ivf_gallery<float, EUCLIDEAN>::add(const float* const x, const int n);

template<typename T,
        DistanceType dist_type>
int ivf_gallery<T, dist_type>::add_one(const int id, const int cq_id, const T* data){
    this->data[cq_id].resize(1LL * (this->block_num[cq_id] + 1) * this->dimension);
    this->ids[cq_id].push_back(id);
    this->index[id] = std::make_pair(cq_id, this->block_num[cq_id]);
    memcpy(this->data[cq_id].data() + (this->block_num[cq_id]) * this->dimension, data, this->dimension);
    this->offset[cq_id].push_back(get_offset<T, dist_type>(this->block_num[cq_id] * this->dimension, this->dimension) );
    this->block_num[cq_id]++;
    this->num++;
    return 0;
}
template int ivf_gallery<int8_t, COSINE>::add_one(const int id, const int cq_id, const T* data);
template int ivf_gallery<float, COSINE>::add_one(const int id, const int cq_id, const T* data);
template int ivf_gallery<int8_t, EUCLIDEAN>::add_one(const int id, const int cq_id, const T* data);
template int ivf_gallery<float, EUCLIDEAN>::add_one(const int id, const int cq_id, const T* data);


template<typename T,
        DistanceType dist_type>
int ivf_gallery<T, dist_type>::add_with_uids(const T* const x, const idx_t * const uids, const int n){
    this->mtx.lock();
    for (int i = 0; i < n; ++i){
        if (this->index.find(uids[i]) != this->index.end()){
            return INDEX_EXISTS;
        }
    }
    pair<Tout, idx_t>* cq_res;
    pair<Tout, idx_t>* pq_res;
    for (int i = 0; i < n; i += this->max_batch){
        int qn = std::min(this->max_batch, n - i);
        memcpy(x_tmp.data(), x + 1LL * i * this->dimension, 1LL * qn * this->dimension * sizeof(T));
        if (is_same_type<T, int8_t>() == true){
            for (int k = 0; k < 1LL * qn * this->dimension; ++k)
                x_tmp[k] += 64;
        }
        divide(x_tmp.data(), x_tmp_div.data(), qn, this->dimension, this->pq_dimension);
        this->cq_mm->mul(x_tmp.data(), this->cq.data(), this->cq_offset.data(), qn, this->cq_num, &cq_res);
        for (int j = 0; j < qn; ++j)
            this->add_one(uids[i + j], cq_res[j].second, x_tmp);
    }
    this->mtx.unlock();
    return 0;
}
template int ivf_gallery<int8_t, COSINE>::add_with_uids(const int8_t* const x, const idx_t * const uids, const int n);
template int ivf_gallery<float, COSINE>::add_with_uids(const float* const x, const idx_t * const uids, const int n);
template int ivf_gallery<int8_t, EUCLIDEAN>::add_with_uids(const int8_t* const x, const idx_t * const uids, const int n);
template int ivf_gallery<float, EUCLIDEAN>::add_with_uids(const float* const x, const idx_t * const uids, const int n);

template <typename T,
        DistanceType dist_type>
int ivf_gallery<T, dist_type>::change_by_uids(const T* const x, const idx_t * const uids, const int n){
    for (int i = 0; i < n; ++i){
        if (this->index.find(uids[i]) == this->index.end())
            return INDEX_NO_FIND;
    }
    this->remove_by_uids(uids, n);
    this->add_with_uids(x, uids, n);
    return 0;
}
template int ivf_gallery<int8_t, COSINE>::change_by_uids(const int8_t* const x, const idx_t * const uids, const int n);
template int ivf_gallery<float, COSINE>::change_by_uids(const float* const x, const idx_t * const uids, const int n);
template int ivf_gallery<int8_t, EUCLIDEAN>::change_by_uids(const int8_t* const x, const idx_t * const uids, const int n);
template int ivf_gallery<float, EUCLIDEAN>::change_by_uids(const float* const x, const idx_t * const uids, const int n);

template<typename T,
        DistanceType dist_type>
int ivf_gallery<T, dist_type>::remove_by_uids(const idx_t* const uids, const int n){
    this->mtx.lock();
    for (int i = 0; i < n; ++i){
        if (this->index.find(uids[i]) == this->index.end())
            return INDEX_NO_FIND;
    }
    for (int i = 0; i < n; ++i){
        pair<int, int> p = this->index[uids[i]];
        memcpy(this->data[p.first].data() + 1LL * p.second * this->dimension, 
               this->data[p.first].data() + 1LL * (this->block_num[p.first] - 1) * this->dimension, this->dimension * sizeof(int));
        this->ids[p.first][p.second] = this->ids[p.first][this->block_num[p.first] - 1];
        this->index[this->ids[p.first][p.second]] = p;
        this->ids[p.first].pop_back();
        this->offset[p.firset].pop_back();
        this->index.erase(uids[i]);
        this->block_num[p.first]--;
        this->data[p.first].resize(1LL * this->block_num[p.first] * this->dimension);
    }
    this->num -= n;
    this->mtx.unlock();
    return 0;
}
template int ivf_gallery<int8_t, COSINE>::remove_by_uids(const idx_t * const uids, const int n);
template int ivf_gallery<float, COSINE>::remove_by_uids(const idx_t * const uids, const int n);
template int ivf_gallery<int8_t, EUCLIDEAN>::remove_by_uids(const idx_t * const uids, const int n);
template int ivf_gallery<float, EUCLIDEAN>::remove_by_uids(const idx_t * const uids, const int n);

template<typename T,
        DistanceType dist_type>
int ivf_gallery<T, dist_type>::query_by_uids(const idx_t* const uids, const int n, T * x){
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
template int ivf_gallery<int8_t, COSINE>::query_by_uids(const idx_t * const uids, const int n, int8_t* x);
template int ivf_gallery<float, COSINE>::query_by_uids(const idx_t * const uids, const int n, float* x);
template int ivf_gallery<int8_t, EUCLIDEAN>::query_by_uids(const idx_t * const uids, const int n, int8_t* x);
template int ivf_gallery<float, EUCLIDEAN>::query_by_uids(const idx_t * const uids, const int n, float* x);

template<typename T,
        DistanceType dist_type>
int ivf_gallery<T, dist_type>::train(const float* data, const int n, const int dimension){
    k_means<float, dist_type>(data, n, this->cq_num, this->dimension, this->cq_float);
    this->have_train_ = true;
    return 0;
}
template int ivf_gallery<int8_t, COSINE>::train(const float* data, const int n, const int dimension);
template int ivf_gallery<float, COSINE>::train(const float* data, const int n, const int dimension);
template int ivf_gallery<int8_t, EUCLIDEAN>::train(const float* data, const int n, const int dimension);
template int ivf_gallery<float, EUCLIDEAN>::train(const float* data, const int n, const int dimension);

template<typename T,
        DistanceType dist_type>
int ivf_gallery<T, dist_type>::store_train_data(std::string file_name){
    std::cout << "write train data" << std::endl;
    ofstream fout(file_name, ofstream::binary);
    int dt = dist_type;
    if (this->have_train_ == false)
        return NO_TRAIN;
    r_write(fout, &this->dimension, 1);
    r_write(fout, &dt, 1);
    r_write(fout, &this->cq_num, 1);
    r_write(fout, &traits, 1);
    r_write(fout, this->cq_float.data(), this->cq_num * this->dimension);
    return 0;
}
template int ivf_gallery<int8_t, COSINE>::store_train_data(std::string file_name);
template int ivf_gallery<float, COSINE>::store_train_data(std::string file_name);
template int ivf_gallery<int8_t, EUCLIDEAN>::store_train_data(std::string file_name);
template int ivf_gallery<float, EUCLIDEAN>::store_train_data(std::string file_name);

template<typename T,
        DistanceType dist_type>
int ivf_gallery<T, dist_type>::load_train_data(std::string file_name){
    std::cout << "read train data" << std::endl;
    struct int cq_num;
    int d, dt;
    ifstream fin(file_name, ifstream::binary);
    r_read(fin, &d, 1);
    r_read(fin, &dt, 1);
    r_read(fin, &cq_num, 1);
    if (d != this->dimension || dt != dist_type || cq_num != this->cq_num)
        return TRAINDATA_ERROR;
    r_read(fin, this->cq_float.data(), this->cq_num * this->dimension);
    this->have_train_ = true;
    return 0;
}
template int ivf_gallery<int8_t, COSINE>::load_train_data(std::string file_name);
template int ivf_gallery<float, COSINE>::load_train_data(std::string file_name);
template int ivf_gallery<int8_t, EUCLIDEAN>::load_train_data(std::string file_name);
template int ivf_gallery<float, EUCLIDEAN>::load_train_data(std::string file_name);


template<typename T,
        DistanceType dist_type>
int ivf_gallery<T, dist_type>::load_data(std::string file_name){
    if (this->have_train_ == false)
        return NO_TRAIN;
    this->mtx.lock();
    std::cout << "[load data] target 1" << std::endl;
    ifstream fin(file_name, ifstream::binary);
    int type, d;
    r_read(fin, &type, 1);
    r_read(fin, &d, 1);
    if (type != ivf_GALLERY || d != this->dimension)
        return LOAD_DATA_ERROR;
    //struct ivf_traits traits;
    int cq_num;
    r_read(fin, &cq_num, 1);
    if (cq_num != this->cq_num)
        return LOAD_DATA_ERROR;
    vector<int> block_num_tmp(this->cq_num);
    vector<idx_t> ids_tmp;
    vector<Tout> offset;
    ids_tmp.resize(this->cq_num);

    r_read(fin, block_num_tmp.data(), this->cq_num);
    std::cout << "[load data] target 2" << std::endl;
    for (int i = 0; i < this->cq_num; ++i){
        ids_tmp.resize(block_num_tmp[i]);
        r_read(fin, ids_tmp.data(), block_num_tmp[i]);
        for (int j = 0; j < block_num_tmp[i]; ++j){
            if (this->index.find(ids_tmp[i][j]) != this->index.end())
                return INDEX_EXISTS;
        }
        memcpy(this->ids[i].data() + this->block_num[i], ids_tmp[i].data(), block_num_tmp[i] * sizeof(idx_t));
    }
    std::cout << "[load data] target 3" << std::endl;
    for (int i = 0; i < cq_num; ++i){
        offset_tmp[i].resize(block_num_tmp[i]);
        this->offset[i].resize(this->block_num[i] + block_num_tmp[i]);
        memcpy(this->offset[i] + this->block_num[i], offset_tmp.data(), block_num_tmp[i] * sizeof());
    }
    for (int i = 0; i < this->cq_num; ++i){
        this->ids[i].resize(this->block_num[i] + block_num_tmp[i]);
        
        for (int j = 0; j < block_num_tmp[i]; ++j)
            this->index[this->ids[i][this->block_num[i] + j]] = std::make_pair(i, this->block_num[i] + j);

        this->data[i].resize(1LL * (this->block_num[i] + block_num_tmp[i]) * this->dimension);
        r_read(fin, this->data[i].data() + 1LL * this->block_num[i] * this->dimension, 1LL * block_num_tmp[i] * this->dimension);
        this->block_num[i] += block_num_tmp[i];
    }
    this->mtx.unlock();
    return 0;
}
template int ivf_gallery<int8_t, COSINE>::load_data(std::string file_name);
template int ivf_gallery<float, COSINE>::load_data(std::string file_name);
template int ivf_gallery<int8_t, EUCLIDEAN>::load_data(std::string file_name);
template int ivf_gallery<float, EUCLIDEAN>::load_data(std::string file_name);

template<typename T,
        DistanceType dist_type>
int ivf_gallery<T, dist_type>::store_data(std::string file_name){
    this->mtx.lock();
    ofstream fout(file_name, ofstream::binary);
    int type = ivf_GALLERY;
    r_write(fout, &type, 1);
    r_write(fout, &this->dimension, 1);
    //struct ivf_traits traits = {this->cq_num, this->select_cq, this->pq_dimension, this->pq_num};
    r_write(fout, &this->cq_num, 1);
    r_write(fout, this->block_num.data(), this->cq_num);
    for (int i = 0; i < this->cq_num; ++i)
        r_write(fout, this->ids[i].data(), this->block_num[i]);
    for (int i = 0; i < this->cq_num; ++i)
        r_write(fout, this->offset[i].data(), 1LL * this->block_num[i]);
    for (int i = 0; i < this->cq_num; ++i)
        r_write(fout, this->data[i].data(), 1LL * this->block_num[i] * this->dimension);
    this->mtx.unlock();
    return 0;
}
template int ivf_gallery<int8_t, COSINE>::store_data(std::string file_name);
template int ivf_gallery<float, COSINE>::store_data(std::string file_name);
template int ivf_gallery<int8_t, EUCLIDEAN>::store_data(std::string file_name);
template int ivf_gallery<float, EUCLIDEAN>::store_data(std::string file_name);
}
