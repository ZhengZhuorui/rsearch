#pragma once
#include <bits/stdc++.h>
#include "rsearch_def.h"
namespace rsearch{

typedef enum CompareType{
    COMP_LT = 0,
    COMP_GT = 1,
    COMP_EQ = 2,
    COMP_LTE = 3,
    COMP_GTE = 5,
}CompareType;
typedef enum CompareResult{
    LT_B = 1,
    EQ_B = 2,
    GT_B = 4,
}CompareResult;

typedef struct query_form{
    int offset;
    char data[16];
    CompareType comp_type;
    VarType var_type;
}query_form;

typedef struct area_time{
    float longtitude, latitude;
    int64_t timestamp; 
}area_time;

template <typename T>
inline query_form set_query_form(int p, char* v, CompareType comp_type){
    query_form form;
    return form;
};

inline area_time construct_area_time(float longtitude, float latitude, int64_t timestamp){
    area_time a;
    a.longtitude = longtitude;
    a.latitude = latitude;
    a.timestamp = timestamp;
    return a;
}

template<>
inline query_form set_query_form<area_time>(int p, char* v, CompareType comp_type){
    query_form form;
    form.comp_type = comp_type;
    switch (p){
        case 0:
            form.offset = 0;
            memcpy(form.data, v, 4);
            form.var_type = FLOAT32;
            break;
        case 1:
            form.offset = 4;
            memcpy(form.data, v, 4);
            form.var_type = FLOAT32;
            break;
        case 2:
            form.offset = 8;
            memcpy(form.data, v, 8);
            form.var_type = INT64;
            break;
        default:
        break;
    }
    return form;
}

inline query_form query_area_time_longtitude_lte(float v){
    return set_query_form<area_time>(0, (char*)&v, COMP_LTE);
}
inline query_form query_area_time_longtitude_gte(float v){
    return set_query_form<area_time>(0, (char*)&v, COMP_GTE);
}
inline query_form query_area_time_latitude_lte(float v){
    return set_query_form<area_time>(1, (char*)&v, COMP_LTE);
}
inline query_form query_area_time_latitude_gte(float v){
    return set_query_form<area_time>(1, (char*)&v, COMP_GTE);
}
inline query_form query_area_time_timestamp_lte(int64_t v){
    return set_query_form<area_time>(2, (char*)&v, COMP_LTE);
}
inline query_form query_area_time_timestamp_gte(int64_t v){
    return set_query_form<area_time>(2, (char*)&v, COMP_GTE);
}

inline int** get_int_pp(){
    return (int**)(malloc(sizeof(int**)));
}

inline int get_int(int *a){
    return *a;
}

inline int* get_int_p(){
    return (int*)(malloc(sizeof(int*)));
}
inline int* get_int_p(int** a){
    return *a;
}

}
