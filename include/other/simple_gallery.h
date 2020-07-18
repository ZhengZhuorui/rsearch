#pragma once
#include "other/other.h"
#include "gallery/rsearch_gallery.h"
namespace rsearch{
template<typename T>
class simple_gallery : public gallery<T>{
public:
    simple_gallery();
    virtual ~simple_gallery() override;
    
    virtual int init() override ; 

    virtual int add(const T * const x, const int n) override ;

    virtual int add_with_uids(const T * const x, const idx_t * const uid, const int n) override ;

    virtual int change_by_uids(const T * const x, const idx_t * const uid, const int n) override ;

    virtual int remove_by_uids(const idx_t * const uid, const int n) override ;

    virtual int query_by_uids(const idx_t * const uid, int n, T * x) override ;

    virtual int reset() override ;

    virtual int store_data(std::string file_name) override;

    virtual int load_data(std::string file_name) override;

    virtual int train(const float* const x, int n){return NO_SUPPORT;}

private:
    idx_t num,max_id;
    std::mutex mtx;

    std::vector<T> data;
    std::vector<idx_t> ids;
    std::unordered_map<idx_t, idx_t> index;
    friend simple_index<T>;

};

}