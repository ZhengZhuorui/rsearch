#include <bits/stdc++.h>
#include <rsearch_type.h>
#include "unit_test.h"
using namespace std;
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
template<> float get_thresh<int>(rsearch::DistanceType dis_type){
    return get_thresh<int8_t>(dis_type);
}

template<> float sim_convert(float v, float scale) {
    return v;    
}

template<> float sim_convert(int v, float scale) {
    return  float(v) / (scale * scale);
}

template<typename T>
bool test_equal(T* sims, uint32_t* uids, uint32_t* real_uids, const int n, const int K, bool sims_equal, bool uids_equal, rsearch::DistanceType dis_type, float scale, bool msg){
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

template bool test_equal(float* sims, uint32_t* uids, uint32_t* real_uids, const int n, const int K, bool sims_equal, bool uids_equal, rsearch::DistanceType dis_type, float scale, bool msg);
template bool test_equal(int* sims, uint32_t* uids, uint32_t* real_uids, const int n, const int K, bool sims_equal, bool uids_equal, rsearch::DistanceType dis_type, float scale, bool msg);

int main(int argc, char* argv[])
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}