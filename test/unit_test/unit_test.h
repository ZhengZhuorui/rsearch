#pragma once
#include <rsearch_type.h>
#include <gtest/gtest.h>
#define private public

const float SCALE = 463.0f;

template<typename T>
float sim_convert(T v, float scale);

template<typename T>
float get_thresh(rsearch::DistanceType dis_type=rsearch::COSINE);

template<typename T>
bool test_equal(T* sims, uint32_t* uids, uint32_t* real_uids, const int n, const int K, bool sims_equal, bool uids_equal, rsearch::DistanceType dis_type, float scale=463.0, bool msg=true);

class UnitTest: public ::testing::Test {
protected:
    static void SetUpTestCase() {
    }

    static void TearDownTestCase() {
    }
protected:

};