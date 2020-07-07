#pragma once
#include "rsearch_gallery.h"
#include <bits/stdc++.h>

namespace rsearch{
using std::vector;
template <typename T,
        DistanceType dist_type>
class ivf_gallery{
public:
    using Tout = typemap_t<T>;
    ivf_gallery(int dimension);

    virtual ~ivf_gallery();

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
    vector<T> cq;
    vector<Tout> cq_offset;
    vector<Tout> cq_float;
    vectorvector<T> > data;
    vector<vector<idx_t> > index;
    vector<pair<idx_t, idx_t> > ids;
    vector<vector<idx_t> > offset;

    int num;
    idx_t max_id;
    int cq_num;
    int max_batch;
    int max_block;

}
}