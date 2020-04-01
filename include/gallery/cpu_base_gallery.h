#pragma once
#include "rsearch_gallery.h"
#include "utils/utils.h"
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

    virtual int store_data(std::string file_name) override;

    virtual int load_data(std::string file_name) override;

private:
    vector<T> data;
    unordered_map<idx_t, idx_t> index;
    vector<idx_t> ids;
    vector<Tout> offset;

    int dimension;
    int num;
    idx_t max_id;
    
    std::mutex mtx;
    friend cpu_base_probe<T, dist_type, base_matrix_mul<T> >;
    friend cpu_base_probe<T, dist_type, rapid_matrix_mul<T> >;
    friend cpu_base_mt_probe<T, dist_type, rapid_matrix_mul<T> >;
};

}