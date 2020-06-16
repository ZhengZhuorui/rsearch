#include <bits/stdc++.h>
#include "unit_test.h"
#include "utils/utils.h"
#include <rsearch_def.h>
#include <gallery/rsearch_gallery.h>
#include <probe/rsearch_probe.h>
#include <utils/utils.h>
#include <sys/time.h>

using namespace std;
using T = float;
using Tout = rsearch::typemap_t<T>;
int n, dimension;
//int n2, dimension2;
int topk = 128;
int need = 16;
int batch = 512;
int target = 11023;
vector<float> vec;
//vector<float> vec2;
vector<Tout> sims;
vector<float> real_sims;
vector<rsearch::idx_t> real_uids, uids;
vector<float> random_data;
int nIter = 1;
/*int get_ans_2(){
    struct timezone zone;
    struct timeval time1;
    struct timeval time2;
    sims.resize(batch * topk);
    uids.resize(batch * topk);
    std::cout << "get ans 2: target 0" << std::endl;
    rsearch::probe<float>* p = rsearch::create_probe<float>(dimension2, topk, rsearch::EUCLIDEAN, rsearch::X86_RAPID_MULTI_THREAD);
    //rsearch::probe<float>* p = rsearch::create_probe<float>(dimension, topk, rsearch::EUCLIDEAN, rsearch::FAISS_HNSW);
    rsearch::gallery<float>* ga;
    p->create_gallery(&ga);
    gettimeofday(&time1, &zone);
    ga->train(vec2.data(), n2);
    //std::cout << "get ans 2: target 1" << std::endl;
    //ga->train(random_data.data(), n);
    std::cout << "get ans 2: target 2" << std::endl;
    if (ga->init() != 0){
        printf("Gallery init failed.\n");
        return -1;
    }
    ga->add(vec2.data(), n2);
    gettimeofday(&time2, &zone);
    float delta = (time2.tv_sec - time1.tv_sec) * 1000.0 + (time2.tv_usec - time1.tv_usec) / 1000.0;
    printf("Gallery Insertion, cost:%4fms\n",  delta);

    std::cout << "get ans 2: target 3" << std::endl;
    gettimeofday(&time1, &zone);
    for (int i = 0; i < nIter; ++i)
        p->query(vec2.data() + target * dimension2, batch, ga, sims.data(), uids.data());
    gettimeofday(&time2, &zone);
    std::cout << "get ans 2: target 4" << std::endl;
    delta = (time2.tv_sec - time1.tv_sec) * 1000.0 + (time2.tv_usec - time1.tv_usec) / 1000.0;
    float QPS = 1000 / (delta / nIter) * batch;
    printf("Batch [%3d] Query On %8d Gallery, cost:%4fms, QPS: %.2f\n", batch, n, delta/nIter, QPS);
    return 0;
}*/
int get_ans_2(){
    struct timezone zone;
    struct timeval time1;
    struct timeval time2;
    sims.resize(batch * topk);
    uids.resize(batch * topk);
    std::cout << "get ans 2: target 0" << std::endl;
    rsearch::probe<T>* p = rsearch::create_probe<T>(dimension, topk, rsearch::EUCLIDEAN, rsearch::X86_RAPID_MULTI_THREAD);
    //rsearch::probe<float>* p = rsearch::create_probe<float>(dimension, topk, rsearch::EUCLIDEAN, rsearch::FAISS_IVFPQ);
    //pair<float, float> args = rsearch::__float_7bits(vec.data(), 1LL * n * dimension);
    //vector<T> vec2(1LL * n * dimension);
    //rsearch::float_7bits(vec.data(), vec2.data(), 1LL * n * dimension, args.first, args.second);
    rsearch::gallery<T>* ga;
    p->create_gallery(&ga);
    gettimeofday(&time1, &zone);
    ga->train(vec.data(), n);
    //std::cout << "get ans 2: target 1" << std::endl;
    //ga->train(random_data.data(), n);
    std::cout << "get ans 2: target 2" << std::endl;
    if (ga->init() != 0){
        printf("Gallery init failed.\n");
        return -1;
    }
    ga->add(vec.data(), n);
    gettimeofday(&time2, &zone);
    float delta = (time2.tv_sec - time1.tv_sec) * 1000.0 + (time2.tv_usec - time1.tv_usec) / 1000.0;
    printf("Gallery Insertion, cost:%4fms\n",  delta);

    std::cout << "get ans 2: target 3" << std::endl;
    gettimeofday(&time1, &zone);
    for (int i = 0; i < nIter; ++i)
        p->query(vec.data() + target * dimension, batch, ga, sims.data(), uids.data());
    gettimeofday(&time2, &zone);
    std::cout << "get ans 2: target 4" << std::endl;
    delta = (time2.tv_sec - time1.tv_sec) * 1000.0 + (time2.tv_usec - time1.tv_usec) / 1000.0;
    float QPS = 1000 / (delta / nIter) * batch;
    printf("Batch [%3d] Query On %8d Gallery, cost:%4fms, QPS: %.2f\n", batch, n, delta/nIter, QPS);
    return 0;
}
int get_ans_1(){
    real_sims.resize(batch * topk);
    real_uids.resize(batch * topk);
    rsearch::probe<float>* p = rsearch::create_probe<float>(dimension, topk, rsearch::EUCLIDEAN, rsearch::X86_RAPID_MULTI_THREAD);
    rsearch::gallery<float>* ga;
    
    p->create_gallery(&ga);
    if (ga->init() != 0){
        printf("Gallery init failed.\n");
        return -1;
    }
    ga->add(vec.data(), n);
    std::cout << "ans_target1" << std::endl;
    p->query(vec.data() + target * dimension, batch, ga, real_sims.data(), real_uids.data());
    for (int i = 0; i < topk; ++i)
        std::cout << real_uids[1 * topk + i] << std::endl;
    std::cout << "ans_target2" << std::endl;
    return 0;
}
int test_compare(){
    
    ifstream fin("/home/zzr/data/1000000x4096", ifstream::binary);
    rsearch::r_file2bytes(fin, vec, n, dimension);
    std::cout << "n=" << n << ", dimension=" << dimension << " " << vec.size()<< std::endl;
    std::cout << vec[0] << " " << vec[1LL * n * dimension-1] << std::endl;
    fin.close();
    /*ifstream fin2("/home/zzr/data/1000000x256", ifstream::binary);
    rsearch::r_file2bytes(fin2, vec2, n2, dimension2);
    std::cout << "n2=" << n2 << ", dimension2=" << dimension2 << " " << vec2.size()<< std::endl;
    std::cout << vec2[0] << " " << vec2[1LL * n2 * dimension2-1] << std::endl;
    fin2.close();*/
    if (get_ans_1() != 0)
        return -1;
    if (get_ans_2() != 0)
        return -1;
    std::cout << "target1" << std::endl;
    
    
    std::cout << "target2" << std::endl;
    int64_t TP= 0, FN, FP, TN;
    float mAP = 0;
    for (int i = 0; i < batch; ++i){
        float AP = 0;
        int cor = 0;
        for (int j = 0; j < need; ++j){
            //std::cout << sims[i * topk + j] << " " << uids[i * topk + j] << " | ";
            int flag = 0;
            for (int k = 0; k < topk; ++k){
                //if (uids[i * topk + j] == real_uids[i * topk + k]) ++TP;
                if (real_uids[i * topk + j] == uids[i * topk + k]) ++cor, ++TP;
                
            }
            AP += cor / (j + 1);
        }
        AP /= need;
        mAP += AP;
    }
    mAP /= batch;
    FN = batch * need - TP;
    FP = batch * topk - TP;
    TN = 1LL * batch * (n-need) - FP;

    std::cout << "TP=" << TP << ", FN=" << FN << ",FP=" << FP << ",TN=" << TN << std::endl;
    std::cout << "accuracy rate: " << 1.0 * (TP + TN) / (batch * n) << std::endl;
    std::cout << "recall rate: " << 1.0 * TP / (TP + FN) << std::endl;
    std::cout << "mAP@128=" << mAP << std::endl;
    return 0;
    //float k = rsearch::__float_7bits(vec.data(), vec_int.data(), n * dimension);
}
TEST_F(UnitTest, QueryCompareTest) {
    EXPECT_EQ(0, test_compare());
}