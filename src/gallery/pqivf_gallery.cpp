#include "gallery/pqivf_gallery.h"
#include "utils/helpers.h"
#include "utils/utils.h"
#include "faiss/Clustering.h"
namespace rsearch{
using std::ofstream;
using std::ifstream;
using std::unordered_map;
using std::vector;
using std::pair;
template<typename T,
        DistanceType dist_type>
pqivf_gallery<T, dist_type>::pqivf_gallery(int dimension, struct pqivf_traits& traits) : gallery<T>(){
    this->dimension = dimension;
    this->num = 0;
    this->max_id = 0;
    this->have_train_ = false;
    this->cq_num = traits.cq_num;
    this->select_cq = traits.select_cq;
    this->pq_num = traits.pq_num;
    this->pq_dimension = traits.pq_dimension;
    this->code_len = this->dimension / this->pq_dimension;
    this->codebook_size = this->code_len * this->pq_num;

    this->max_batch = 32;
    this->max_block = 51200;
    
    this->cq.resize(this->cq_num * this->dimension);
    this->cq_offset.resize(this->cq_num);
    this->cq_float.resize(this->cq_num * this->dimension);

    this->pq.resize(this->codebook_size * this->pq_dimension);
    this->pq_offset.resize(this->codebook_size);
    this->pq_float.resize(this->codebook_size * this->pq_dimension);

    this->block_num.resize(this->cq_num);
    this->ids.resize(this->cq_num);
    this->data.resize(this->cq_num);
    this->x_tmp.resize(this->max_batch * this->dimension);
    this->x_tmp_div.resize(this->max_batch * this->dimension);
    this->index.clear();
    this->cq_mm = new rapid_matrix_mul<T>();
    this->pq_mm = new rapid_matrix_mul<T>();
    this->cq_mm->set(this->dimension, 1, this->max_batch, this->cq_num);
    this->pq_mm->set(this->pq_dimension, 1, this->max_batch, this->pq_num);

}
template pqivf_gallery<int8_t, COSINE>::pqivf_gallery(int, struct pqivf_traits&);
template pqivf_gallery<float, COSINE>::pqivf_gallery(int, struct pqivf_traits&);
template pqivf_gallery<int8_t, EUCLIDEAN>::pqivf_gallery(int, struct pqivf_traits&);
template pqivf_gallery<float, EUCLIDEAN>::pqivf_gallery(int, struct pqivf_traits&);

template<typename T,
        DistanceType dist_type>
pqivf_gallery<T, dist_type>::~pqivf_gallery(){
    delete this->cq_mm;
    delete this->pq_mm;
}
template pqivf_gallery<int8_t, COSINE>::~pqivf_gallery();
template pqivf_gallery<float, COSINE>::~pqivf_gallery();
template pqivf_gallery<int8_t, EUCLIDEAN>::~pqivf_gallery();
template pqivf_gallery<float, EUCLIDEAN>::~pqivf_gallery();

template<typename T,
        DistanceType dist_type>
int pqivf_gallery<T, dist_type>::init(){
    std::string dist_type_name = GetDistancetypeName<dist_type>();
    std::string train_file_name = std::string(DATA_DIR) + "/pqivf_train_data." + dist_type_name + "." + std::to_string(this->dimension)+ "." + \
        std::to_string(this->cq_num) + "." + std::to_string(this->pq_dimension) + "." + std::to_string(this->pq_num) + ".bin";
    std::cout << train_file_name << std::endl;
    #ifdef DEBUG_0
    if (this->have_train_ == false){
        if (file_exist(train_file_name) == true){
            std::cout << train_file_name << std::endl;
            int code = this->load_train_data(train_file_name);
            if (code == TRAINDATA_ERROR)
                return TRAINDATA_ERROR;
        }
        else{
            vector<float> data;
            std::cout << "[init] target 2" <<std::endl;
            get_random_data<float, dist_type>(data, 200000, this->dimension);
            this->train(data.data(), 200000);
            this->store_train_data(train_file_name);            
        }
    }
    #else
    if (this->have_train_ == false){
        return NO_TRAIN;
    }
    #endif
    if (is_same_type<T, int8_t>() == true){
        pair<float, float> arg = __float_7bits(this->cq_float.data(), this->cq_num * this->dimension);
        float_7bits(this->cq_float.data(), (int8_t*)this->cq.data(), this->cq_num * this->dimension, arg.first, arg.second);
        arg = __float_7bits(this->pq_float.data(), this->pq_num * this->dimension);
        //std::cout << "arg:" << arg.first << " " << arg.second << std::endl;
        float_7bits(this->pq_float.data(), (int8_t*)this->pq.data(), this->pq_num * this->dimension, arg.first, arg.second);
    }
    else{
        memcpy(this->cq.data(), this->cq_float.data(), this->cq_num * this->dimension * sizeof(T) );
        memcpy(this->pq.data(), this->pq_float.data(), this->codebook_size * this->pq_dimension * sizeof(T) );
    }
    this->cq_float.clear();
    this->pq_float.clear();
    std::cout << "[init] target 2" << std::endl;
    for (int i = 0; i < this->cq_num; ++i){
        this->cq_offset[i] = get_offset<T, dist_type>(this->cq.data() + 1LL * i * this->dimension, this->dimension);
    }
    for (int i = 0; i < this->pq_num * this->code_len; ++i){
        this->pq_offset[i] = get_offset<T, dist_type>(this->pq.data() + 1LL * i * this->pq_dimension, this->pq_dimension);
    }
    return 0;
}
template int pqivf_gallery<int8_t, COSINE>::init();
template int pqivf_gallery<float, COSINE>::init();
template int pqivf_gallery<int8_t, EUCLIDEAN>::init();
template int pqivf_gallery<float, EUCLIDEAN>::init();

template<typename T,
        DistanceType dist_type>
int pqivf_gallery<T, dist_type>::reset(){
    this->index.clear();
    this->cq.clear();
    this->pq.clear();
    this->cq_offset.clear();
    this->pq_offset.clear();
    this->cq_float.clear();
    this->pq_float.clear();
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
template int pqivf_gallery<int8_t, COSINE>::reset();
template int pqivf_gallery<float, COSINE>::reset();
template int pqivf_gallery<int8_t, EUCLIDEAN>::reset();
template int pqivf_gallery<float, EUCLIDEAN>::reset();

template<typename T,
        DistanceType dist_type>
int pqivf_gallery<T, dist_type>::add(const T* const x, const int n){
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
        divide(x_tmp.data(), x_tmp_div.data(), qn, this->dimension, this->pq_dimension);
        this->cq_mm->mul(x_tmp.data(), this->cq.data(), this->cq_offset.data(), qn, this->cq_num, &cq_res);
        for (int j = 0; j < qn; ++j)
            this->add_one(this->max_id++, cq_res[j].second);
        
        for (int j = 0; j < this->code_len; ++j){
            this->pq_mm->mul(x_tmp_div.data() + j * qn * this->pq_dimension, this->pq.data() + j * this->pq_num * this->pq_dimension,
                            this->pq_offset.data() + j * this->pq_num,
                            qn, this->pq_num, &pq_res);
            for (int k = 0; k < qn; ++k){
                std::pair<int, int> p = this->index[this->max_id - qn + k];
                this->data[p.first][p.second * this->code_len + j] = pq_res[k].second + j * this->pq_num;
            }
        }
    }
    //std::cout << "[add]" << cq_res[0].second << std::endl;
    this->mtx.unlock();
    return 0;
}
template int pqivf_gallery<int8_t, COSINE>::add(const int8_t* const x, const int n);
template int pqivf_gallery<float, COSINE>::add(const float* const x, const int n);
template int pqivf_gallery<int8_t, EUCLIDEAN>::add(const int8_t* const x, const int n);
template int pqivf_gallery<float, EUCLIDEAN>::add(const float* const x, const int n);

template<typename T,
        DistanceType dist_type>
int pqivf_gallery<T, dist_type>::add_one(const int id, const int cq_id){
    this->data[cq_id].resize(1LL * (this->block_num[cq_id] + 1) * this->code_len);
    //memcpy(this->data[cq_id].data() + 1LL * this->block_num[cq_id] * this->code_len, x, this->code_len * sizeof(uint8_t));
    //for (int j = 0; j < this->code_len; ++j)
    //    this->data[cq_id][1LL * this->block_num[cq_id] * this->code_len + j] = x[j].second + j * this->pq_num;
    this->ids[cq_id].push_back(id);
    this->index[id] = std::make_pair(cq_id, this->block_num[cq_id]);
    this->block_num[cq_id]++;
    this->num++;
    return 0;
}
template int pqivf_gallery<int8_t, COSINE>::add_one(const int id, const int cq_id);
template int pqivf_gallery<float, COSINE>::add_one(const int id, const int cq_id);
template int pqivf_gallery<int8_t, EUCLIDEAN>::add_one(const int id, const int cq_id);
template int pqivf_gallery<float, EUCLIDEAN>::add_one(const int id, const int cq_id);


template<typename T,
        DistanceType dist_type>
int pqivf_gallery<T, dist_type>::add_with_uids(const T* const x, const idx_t * const uids, const int n){
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
            this->add_one(uids[i + j], cq_res[j].second);
        for (int j = 0; j < this->code_len; ++j){
            this->pq_mm->mul(x_tmp_div.data() + j * qn * this->pq_dimension, this->pq.data() + j * this->pq_num * this->pq_dimension,
                            this->pq_offset.data() + j * this->pq_num,
                            qn, this->pq_num, &pq_res);
            for (int k = 0; k < qn; ++k){
                std::pair<int, int> p = this->index[uids[i + k]];
                this->data[p.first][p.second * this->code_len + j] = pq_res[k].second + j * this->pq_num;
            }
        }
    }
    this->mtx.unlock();
    return 0;
}
template int pqivf_gallery<int8_t, COSINE>::add_with_uids(const int8_t* const x, const idx_t * const uids, const int n);
template int pqivf_gallery<float, COSINE>::add_with_uids(const float* const x, const idx_t * const uids, const int n);
template int pqivf_gallery<int8_t, EUCLIDEAN>::add_with_uids(const int8_t* const x, const idx_t * const uids, const int n);
template int pqivf_gallery<float, EUCLIDEAN>::add_with_uids(const float* const x, const idx_t * const uids, const int n);

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
template int pqivf_gallery<int8_t, COSINE>::change_by_uids(const int8_t* const x, const idx_t * const uids, const int n);
template int pqivf_gallery<float, COSINE>::change_by_uids(const float* const x, const idx_t * const uids, const int n);
template int pqivf_gallery<int8_t, EUCLIDEAN>::change_by_uids(const int8_t* const x, const idx_t * const uids, const int n);
template int pqivf_gallery<float, EUCLIDEAN>::change_by_uids(const float* const x, const idx_t * const uids, const int n);

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
               this->data[p.first].data() + 1LL * (this->block_num[p.first] - 1) * this->code_len, this->code_len * sizeof(int));
        this->ids[p.first][p.second] = this->ids[p.first][this->block_num[p.first] - 1];
        this->index[this->ids[p.first][p.second]] = p;
        this->ids[p.first].pop_back();
        this->index.erase(uids[i]);
        this->block_num[p.first]--;
        this->data[p.first].resize(1LL * this->block_num[p.first] * this->code_len);
    }
    this->num -= n;
    this->mtx.unlock();
    return 0;
}
template int pqivf_gallery<int8_t, COSINE>::remove_by_uids(const idx_t * const uids, const int n);
template int pqivf_gallery<float, COSINE>::remove_by_uids(const idx_t * const uids, const int n);
template int pqivf_gallery<int8_t, EUCLIDEAN>::remove_by_uids(const idx_t * const uids, const int n);
template int pqivf_gallery<float, EUCLIDEAN>::remove_by_uids(const idx_t * const uids, const int n);

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
template int pqivf_gallery<int8_t, COSINE>::query_by_uids(const idx_t * const uids, const int n, int8_t* x);
template int pqivf_gallery<float, COSINE>::query_by_uids(const idx_t * const uids, const int n, float* x);
template int pqivf_gallery<int8_t, EUCLIDEAN>::query_by_uids(const idx_t * const uids, const int n, int8_t* x);
template int pqivf_gallery<float, EUCLIDEAN>::query_by_uids(const idx_t * const uids, const int n, float* x);

template<typename T,
        DistanceType dist_type>
int pqivf_gallery<T, dist_type>::train(const float* data, const int n){
    //k_means<float, dist_type>(data, n, this->cq_num, this->dimension, this->cq_float);
    faiss::kmeans_clustering(this->dimension, n, this->cq_num, data, this->cq_float.data());

    vector<float> data_tmp(1LL * n * this->dimension);
    divide(data, data_tmp.data(), n, dimension, pq_dimension);
    for (int i = 0; i < this->code_len; ++i){
        std::cout << "train: " << i << std::endl;
        vector<float> pq_tmp(1LL * this->pq_num * this->pq_dimension);
        //k_means<float, dist_type>(data_tmp.data() + i * (n * this->pq_dimension), n, this->pq_num, this->pq_dimension, pq_tmp);
        faiss::kmeans_clustering(this->pq_dimension, n, this->pq_num, data_tmp.data() + 1LL * i * n * this->pq_dimension, pq_tmp.data());
        memcpy(this->pq_float.data() + 1LL * i * this->pq_num * this->pq_dimension, pq_tmp.data(), 1LL * sizeof(float) * this->pq_num * this->pq_dimension);
    }
    this->have_train_ = true;
    return 0;
}
template int pqivf_gallery<int8_t, COSINE>::train(const float* data, const int n);
template int pqivf_gallery<float, COSINE>::train(const float* data, const int n);
template int pqivf_gallery<int8_t, EUCLIDEAN>::train(const float* data, const int n);
template int pqivf_gallery<float, EUCLIDEAN>::train(const float* data, const int n);

template<typename T,
        DistanceType dist_type>
int pqivf_gallery<T, dist_type>::store_train_data(std::string file_name){
    std::cout << "write train data" << std::endl;
    ofstream fout(file_name, ofstream::binary);
    int dt = dist_type;
    if (this->have_train_ == false)
        return NO_TRAIN;
    r_write(fout, &this->dimension, 1);

    r_write(fout, &dt, 1);
    struct pqivf_traits traits = {this->cq_num, this->select_cq, this->pq_dimension, this->pq_num};
    r_write(fout, &traits, 1);
    r_write(fout, this->cq_float.data(), this->cq_num * this->dimension);
    r_write(fout, this->pq_float.data(), this->codebook_size * this->pq_dimension);
    return 0;
}
template int pqivf_gallery<int8_t, COSINE>::store_train_data(std::string file_name);
template int pqivf_gallery<float, COSINE>::store_train_data(std::string file_name);
template int pqivf_gallery<int8_t, EUCLIDEAN>::store_train_data(std::string file_name);
template int pqivf_gallery<float, EUCLIDEAN>::store_train_data(std::string file_name);

template<typename T,
        DistanceType dist_type>
int pqivf_gallery<T, dist_type>::load_train_data(std::string file_name){
    std::cout << "read train data" << std::endl;
    struct pqivf_traits traits;
    int d, dt;
    ifstream fin(file_name, ifstream::binary);
    r_read(fin, &d, 1);
    r_read(fin, &dt, 1);
    r_read(fin, &traits, 1);
    if (d != this->dimension || dt != dist_type || traits.cq_num != this->cq_num || 
        traits.pq_dimension != this->pq_dimension || traits.pq_num != this->pq_num)
    return TRAINDATA_ERROR;
    r_read(fin, this->cq_float.data(), this->cq_num * this->dimension);
    r_read(fin, this->pq_float.data(), this->codebook_size * this->pq_dimension);
    this->have_train_ = true;
    return 0;
}
template int pqivf_gallery<int8_t, COSINE>::load_train_data(std::string file_name);
template int pqivf_gallery<float, COSINE>::load_train_data(std::string file_name);
template int pqivf_gallery<int8_t, EUCLIDEAN>::load_train_data(std::string file_name);
template int pqivf_gallery<float, EUCLIDEAN>::load_train_data(std::string file_name);


template<typename T,
        DistanceType dist_type>
int pqivf_gallery<T, dist_type>::load_data(std::string file_name){
    if (this->have_train_ == false)
        return NO_TRAIN;
    this->mtx.lock();
    ifstream fin(file_name, ifstream::binary);
    int type, d;
    r_read(fin, &type, 1);
    r_read(fin, &d, 1);
    if (type != PQIVF_GALLERY || d != this->dimension)
        return LOAD_DATA_ERROR;
    struct pqivf_traits traits;
    r_read(fin, &traits, 1);
    if (traits.cq_num != this->cq_num || traits.select_cq != this->select_cq || traits.pq_dimension != this->pq_dimension ||
        traits.pq_num != this->pq_num)
        return LOAD_DATA_ERROR;
    r_read(fin, &this->max_id, 1);
    vector<int> block_num_tmp(this->cq_num);
    vector<vector<idx_t> > ids_tmp;
    ids_tmp.resize(this->cq_num);
    r_read(fin, block_num_tmp.data(), this->cq_num);
    for (int i = 0; i < this->cq_num; ++i){
        ids_tmp[i].resize(block_num_tmp[i]);
        r_read(fin, ids_tmp[i].data(), block_num_tmp[i]);
        for (int j = 0; j < block_num_tmp[i]; ++j){
            if (this->index.find(ids_tmp[i][j]) != this->index.end())
                return INDEX_EXISTS;
        }
    }
    for (int i = 0; i < this->cq_num; ++i){
        this->ids[i].resize(this->block_num[i] + block_num_tmp[i]);
        memcpy(this->ids[i].data() + this->block_num[i], ids_tmp[i].data(), block_num_tmp[i] * sizeof(idx_t));
        for (int j = 0; j < block_num_tmp[i]; ++j)
            this->index[this->ids[i][this->block_num[i] + j]] = std::make_pair(i, this->block_num[i] + j);
    }
    for (int i = 0; i < this->cq_num; ++i){
        this->data[i].resize(1LL * (this->block_num[i] + block_num_tmp[i]) * this->code_len);
        r_read(fin, this->data[i].data() + 1LL * this->block_num[i] * this->code_len, 1LL * block_num_tmp[i] * this->code_len);
        this->block_num[i] += block_num_tmp[i];
    }
    this->mtx.unlock();
    return 0;
}
template int pqivf_gallery<int8_t, COSINE>::load_data(std::string file_name);
template int pqivf_gallery<float, COSINE>::load_data(std::string file_name);
template int pqivf_gallery<int8_t, EUCLIDEAN>::load_data(std::string file_name);
template int pqivf_gallery<float, EUCLIDEAN>::load_data(std::string file_name);

template<typename T,
        DistanceType dist_type>
int pqivf_gallery<T, dist_type>::store_data(std::string file_name){
    this->mtx.lock();
    ofstream fout(file_name, ofstream::binary);
    int type = PQIVF_GALLERY;
    r_write(fout, &type, 1);
    r_write(fout, &this->dimension, 1);
    struct pqivf_traits traits = {this->cq_num, this->select_cq, this->pq_dimension, this->pq_num};
    r_write(fout, &traits, 1);
    r_write(fout, &this->max_id, 1);
    r_write(fout, this->block_num.data(), this->cq_num);
    for (int i = 0; i < this->cq_num; ++i)
        r_write(fout, this->ids[i].data(), this->block_num[i]);
    for (int i = 0; i < this->cq_num; ++i)
        r_write(fout, this->data[i].data(), 1LL * this->block_num[i] * this->code_len);   
    this->mtx.unlock();
    return 0;
}
template int pqivf_gallery<int8_t, COSINE>::store_data(std::string file_name);
template int pqivf_gallery<float, COSINE>::store_data(std::string file_name);
template int pqivf_gallery<int8_t, EUCLIDEAN>::store_data(std::string file_name);
template int pqivf_gallery<float, EUCLIDEAN>::store_data(std::string file_name);
}
