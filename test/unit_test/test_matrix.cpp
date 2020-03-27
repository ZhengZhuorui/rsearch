#include "unit_test.h"
#include <matrix/rapid_matrix_mul.h>

template<typename T>
int test_topk(int batch, int block, int dimension, int topk){
    /*
    std::vector<T> data(batch * block);
    std::vector<T> data_tmp(batch * block);
    std::vector<T> data_out(batch * topk);
    std::vector<T> index_out(batch * topk);
    matrix<T>* mm = new rapid_matrix_mul<T>;
    mm->set(dimension, topk, batch, block);
    for (int i = 0; i < batch; ++i)
    for (int j = 0 ; j < block; ++j)
        data_tmp[i * block + j] = data[i * block + j] = rand();
    this->mm->mul(data_tmp,)
    

    int res = 0;
    for (int i = 0; i < batch; ++i){
        for (int j = 0; j < topk; ++j)
            if (data_out[i * topk + j] != data_tmp[i * block + j]){
                res = -1;
                std::cout << "Diff [" << i << "][" << j << "]:" << data_tmp[i * block + j]  << ":" << data_out[i * topk + j]<< std::endl; 
            }
    }
    return res;*/
    return 0;
}
/*
TEST_F(UnitTest, TopkTest) {
    EXPECT_EQ(0, test_topk<float>(16, 30000, 512, 128) );
    
}
*/