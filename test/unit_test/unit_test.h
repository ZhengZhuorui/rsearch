#pragma once
#include <gtest/gtest.h>
#include <rsearch_type.h>

const float SCALE = 463.0f;

template<typename T>
float sim_convert(T v, float scale);

template<typename T>
float get_thresh(rsearch::DistanceType dis_type=rsearch::COSINE);

template<> 
float get_thresh<float>(rsearch::DistanceType dis_type) {
    if (dis_type == rsearch::COSINE) {
        return 0.99f;
    } else {
        return 0.01f;
    }
}

template<> float get_thresh<int8_t>(rsearch::DistanceType dis_type) {
    if (dis_type == rsearch::COSINE) {
        return 0.95f;
    } else {
        return 0.05f;
    }
}

template<> float sim_convert(float v, float scale) {
    return v;    
}

template<> float sim_convert(int v, float scale) {
    return  float(v) / (scale * scale);
}

template<typename T>
bool test_equal(T* sims, int* uids, int* real_uids, int n, int K, bool sims_equal, bool uids_equal, rsearch::DistanceType dis_type, float scale, bool msg){
    bool flag;
    if (dis_type == rsearch::COSINE){
        for (int i = 0 ; i < n ; ++i){
            flag = (sim_convert(sims[i*K], scale) > get_thresh<T>(dis_type));
            if (flag != sims_equal){
                if (msg)
                    printf("batch:%d Expected sim:%.7f got:%.7f\n", i, get_thresh<T>(dis_type), sim_convert(sims[i*K], SCALE));
                return false;
            }
            flag = (uids[i*K] == real_uids[i]);
            if (flag != uids_equal){
                if (msg)
                    printf("batch:%d Expected uids:%d got:%d\n", i, real_uids[i], uids[i*K]);
                return false;
            }
        }
    }
    else{
        for (int i = 0 ; i < n ; ++i){
            bool flag = (sim_convert(sims[i*K], scale) < get_thresh<T>(dis_type));
            if (flag != sims_equal){
                if (msg)
                    printf("batch:%d Expected sim:%.7f got:%.7f\n", i, get_thresh<T>(dis_type), sim_convert(sims[i*K], SCALE));
                return false;
            }
            flag = (uids[i*K] == real_uids[i]);
            if (flag != uids_equal){
                if (msg)
                    printf("batch:%d Expected uids:%d got:%d\n", i, real_uids[i], uids[i*K]);
                return false;
            }
        }
    }
    return true;
}



class UnitTest: public ::testing::Test {
protected:
    static void SetUpTestCase() {
    
    }

    static void TearDownTestCase() {
        
    }
protected:

};