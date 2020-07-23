#include "other/simple_index.h"

namespace rsearch{

template <typename T>
int simple_index<T>::add(const T* x, int n){
    return ga->add(x, n);
}

template <typename T>
int simple_index<T>::add_with_uids(const T * x, const idx_t * uids, const int n){
    return ga->add_with_uids(x, uids, n);
}

template <typename T>
int simple_index<T>::change_by_uids(const T * x, const idx_t * uids, const int n){
    return ga->change_by_uids(x, uids, n);
}
template <typename T>
int simple_index<T>::remove_by_uids(const idx_t *  uid, const int n) {
    return ga->remove_by_uids(uid, n);
}
template <typename T>
int simple_index<T>::query_by_uids(const idx_t *  uid, int n, T * x) {
    return ga->query_by_uids(uid, n, x);
}
template <typename T>
int simple_index<T>::reset(){
    return ga->reset();
}
template <typename T>
int simple_index<T>::store_data(const char* file_name){
    return ga->store_data(std::string(file_name));
}
template <typename T>
int simple_index<T>::load_data(const char* file_name){
    return ga->load_data(std::string(file_name));
}

static char compare_by_type(const char *x, const char *y, VarType v){
    switch (v)
    {
    case FLOAT32:
        //std::cout << (float)(*x) << " " << (float)(*y) << std::endl;
        //printf("%f %f\n", (float)(*(float*)x), (float)(*(float*)y));
        if ((float)(*(float*)x) < (float)(*(float*)y)) return LT_B;
        else if ((float)(*(float*)x) == (float)(*(float*)y)) return EQ_B;
        else return GT_B;
        break;
    case INT8:
        if ((char)(*x) < (char)(*y)) return LT_B;
        else if ((char)(*x) == (char)(*y)) return EQ_B;
        else return GT_B;
        break;
    case INT32:
        if ((int32_t)(*(int32_t*)x) < (int32_t)(*(int32_t*)x)) return LT_B;
        else if ((int32_t)(*(int32_t*)x) == (int32_t)(*(int32_t*)x)) return EQ_B;
        else return GT_B;
        break;
    case INT64:
        if ((int64_t)(*x) < (int64_t)(*y)) return LT_B;
        else if ((int64_t)(*x) == (int64_t)(*y)) return EQ_B;
        else return GT_B;
        break;
    default:
        break;
    }
    return 0;
}
template <typename T>
int simple_index<T>::query(const query_form * x, const int n, idx_t** idx, int* res){
    simple_gallery<T>* ga_ptr = this->ga;
    int num = ga_ptr->num;
    T* data = ga_ptr->data.data();
    idx_t* ids = ga_ptr->ids.data();
    std::vector<idx_t>idx_vec;
    idx_vec.clear();
    //std::cout <<  "t1 " << x[0].comp_type << " " << n << std::endl;
    for (int i = 0; i < num; ++i){    
        //if (x.select[i](data[i]) == true) idx.push_back(ids[i]);
        bool flag = true;
        for (int j = 0; j < n; ++j){
            int8_t v = compare_by_type((char*)(&data[i]) + x[j].offset, x[j].data, x[j].var_type);
            //std::cout << "t2" << std::endl;
            switch (x[j].comp_type)
            {
            case COMP_LT:
                if (((v) & LT_B) == 0) flag = false;
                break;
            case COMP_GT:
                if (((v) & GT_B) == 0) flag = false;
                break;
            case COMP_EQ:
                if (((v) & EQ_B) == 0) flag = false;
                break;
            case COMP_LTE:
                if ((((v) & LT_B) == 0) && (((v) & EQ_B) == 0)) flag = false;
                break;
            case COMP_GTE:
                if ((((v) & GT_B) == 0) && (((v) & EQ_B) == 0)) flag = false;
                break;
            default:
                break;
            }
            //std::cout << flag << std::endl;
            if (flag == false) break;
        }
        if (flag == true) idx_vec.push_back(ids[i]);
    }
    //std::cout <<  "t2 " << idx_vec.size() << std::endl;
    (*idx) = (idx_t*)malloc(idx_vec.size() * sizeof(idx_t));
    memcpy((*idx), idx_vec.data(), idx_vec.size() * sizeof(idx_t));
    *res = idx_vec.size();
    return 0;
}
template<typename T>
int simple_index<T>::query_with_uids(const query_form* x, const int n, idx_t *uids, const int m, idx_t** idx, int* res){
    simple_gallery<T>* ga_ptr = this->ga;
    
    T* data = ga_ptr->data.data();
    std::vector<idx_t> idx_vec;
    idx_vec.clear();
    //idx_t* ids = ga_ptr->ids.data();
    std::unordered_map<idx_t, idx_t>& index= ga_ptr->index;
    for (int i = 0; i < m; ++i){
        if (uids[i] == -1){
             continue;
        }
        if (index.find(uids[i]) == index.end()){
            return INDEX_NO_FIND;
        }
        T* select_data = data + index[uids[i]];
        bool flag = true;
        for (int j = 0; j < n; ++j){
            int8_t v = compare_by_type((char*)(select_data) + x[j].offset, x[j].data, x[j].var_type);
            //std::cout<< v << std::endl;
            switch (x[j].comp_type)
            {
            case COMP_LT:
                if (((v) & LT_B) == 0) flag = false;
                break;
            case COMP_GT:
                if (((v) & GT_B) == 0) flag = false;
                break;
            case COMP_EQ:
                if (((v) & EQ_B) == 0) flag = false;
                break;
            case COMP_LTE:
                if ((((v) & LT_B) == 0) && (((v) & EQ_B) == 0)) flag = false;
                break;
            case COMP_GTE:
                if ((((v) & GT_B) == 0) && (((v) & EQ_B) == 0)) flag = false;
                break;
            default:
                break;
            }
            if (flag == false) break;
        }
        if (flag == true) idx_vec.push_back(uids[i]);
    }    
    (*idx) = (idx_t*)malloc(idx_vec.size() * sizeof(idx_t));
    memcpy((*idx), idx_vec.data(), idx_vec.size() * sizeof(idx_t));
    *res = idx_vec.size();
    return 0;
}

// instantiation

template simple_index<area_time>::simple_index();
template simple_index<area_time>::~simple_index();
template int simple_index<area_time>::add(const area_time * x, const int n);
template int simple_index<area_time>::add_with_uids(const area_time * x, const idx_t * const uids, const int n);
template int simple_index<area_time>::change_by_uids(const area_time * x, const idx_t * const uids, const int n);
template int simple_index<area_time>::remove_by_uids(const idx_t * uid, const int n);
template int simple_index<area_time>::query_by_uids(const idx_t * uid, int n, area_time * x);
template int simple_index<area_time>::reset();
template int simple_index<area_time>::store_data(const char* file_name);
template int simple_index<area_time>::load_data(const char* file_name);
template int simple_index<area_time>::query(const query_form * x, const int n, idx_t** idx, int* res);
template int simple_index<area_time>::query_with_uids(const query_form* x, const int n, idx_t *uids, const int m, idx_t** idx, int* res);
}
