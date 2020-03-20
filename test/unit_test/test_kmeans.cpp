#include <bits/stdc++.h>
#include <utils/cluster.h>
#include <utils/utils.h>
#include "unit_test.h"

int test_kmeans(const int N, const int K, const int C){
    std::vector<float> data;
    rsearch::get_random_data<float, rsearch::EUCLIDEAN>(data, N, K);
    std::vector<float> res;
    return rsearch::k_means<float, rsearch::EUCLIDEAN>(data.data(), N, 4096, K, res);
}
TEST_F(UnitTest, KmeansTest) {
    EXPECT_EQ(0, (test_kmeans(200000, 512, 4096)) );
}