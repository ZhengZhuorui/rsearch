#include "unit_test.h"
#include "utils/topk_select.h"
int test_topk(int batch, int block, int topk, int ldc){

    std::vector<int> data(batch * ldc);
    std::vector<int> data_tmp(batch * block);
    std::vector<int> data_out(batch * topk);
    std::vector<int> index_out(batch * topk);
    for (int i = 0; i < batch; ++i)
    for (int j = 0 ; j < block; ++j)
        data_tmp[i * block + j] = data[i * ldc + j] = rand();
    rsearch::cpu_select_kv<int>(data.data(), data_out.data(), index_out.data(), topk, block, batch, ldc, false);
    for (int i = 0; i < batch; ++i)
        std::sort(data_tmp.data() + i * block, data_tmp.data() + (i + 1) * block);
    int res = 0;
    for (int i = 0; i < batch; ++i){
        for (int j = 0; j < topk; ++j)
            if (data_out[i * topk + j] != data_tmp[i * block + j]){
                res = -1;
                std::cout << "Diff [" << i << "][" << j << "]:" << data_tmp[i * block + j]  << ":" << data_out[i * topk + j]<< std::endl; 
            }
    }
    return res;
}

TEST_F(UnitTest, TopkTest) {
    EXPECT_EQ(0, test_topk(10, 30000, 128, 50000) );
    
}
