#pragma once
#include "utils/cluster.h"
#include "rsearch_gallery.h"
#include <bits/stdc++.h>

namespace rsearch{
using std::vector;
using std::unordered_map;
struct pqivf_traits{
    int cq_num;
    int select_cq;
    int pq_dimension;
    int pq_num;
    //float k;
};

template<typename T,
        DistanceType dist_type>
class pqivf_gallery : public gallery<T>{
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

    virtual int store_data(std::string file_name) override;

    virtual int load_data(std::string file_name) override;


    virtual int train(const float* const x, int n, int dimension);

    virtual bool have_train(){return this->have_train_;}

    virtual int store_train_data(std::string file_name);

    virtual int load_train_data(std::string file_name);
    

private:

    int add_one(const pair<Tout, idx_t>* const x, const int id, const int cq_id);

    vector<T> cq;
    vector<T> pq;
    vector<Tout> cq_offset;
    vector<Tout> pq_offset;
    vector<float> cq_float;
    vector<float> pq_float;

    vector<vector<int> > data;
    vector<int> block_num;
    unordered_map<idx_t, pair<int, int> > index;
    vector<vector<idx_t> > ids;
    vector<T> x_tmp;

    int dimension;
    int num;
    idx_t max_id;
    bool have_train_;

    int cq_num;
    int select_cq;
    int pq_dimension;
    int pq_num;
    int code_len;
    int max_batch, max_block;
    std::mutex mtx;

    rapid_matrix_mul<T>* cq_mm;
    rapid_matrix_mul<T>* pq_mm;

    friend pqivf_probe<T, dist_type>;
};


}
