#pragma once
#include "other/other.h"
#include "other/simple_gallery.h"

namespace rsearch{

typedef struct{
    std::vector<idx_t> v;
}result;

template<typename T>
class simple_index{
public:
    simple_index(){
        ga = new simple_gallery<T>();
    }
    ~simple_index(){
        delete ga;
    }
    int add(const T * x, const int n);

    int add_with_uids(const T * x, const idx_t * uids, const int n);

    int change_by_uids(const T * x, const idx_t * uids, const int n);

    int remove_by_uids(const idx_t *  uid, const int n);

    int query_by_uids(const idx_t *  uid, int n, T * x);

    int reset();

    int store_data(std::string file_name);

    int load_data(std::string file_name);

    int query(const query_form *  x, const int n, idx_t** idx, int* res);
    int query_with_uids(const query_form*  x, const int n, idx_t *uids, const int m, idx_t** idx, int* res);
private:
    simple_gallery<T>* ga;
};
}