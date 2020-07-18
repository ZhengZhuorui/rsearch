#include <rsearch_type.h>
#include <gallery/rsearch_gallery.h>
#include <probe/rsearch_probe.h>
#include <utils/utils.h>
#include <utils/helpers.h>
#include <sys/time.h>
#include <other/other.h>
#include <other/simple_index.h>
#include "unit_test.h"

int test_simple_index(){
    rsearch::simple_index<rsearch::area_time>* index = new rsearch::simple_index<rsearch::area_time>();
    std::vector<rsearch::area_time> data;
    data.push_back(rsearch::construct_area_time(0.5, 1.5, 10));
    data.push_back(rsearch::construct_area_time(-0.5, 0.5, 11));
    data.push_back(rsearch::construct_area_time(0.8, 1.0, 12));
    data.push_back(rsearch::construct_area_time(0.2, 0.6, 13));
    data.push_back(rsearch::construct_area_time(0.6, -2.0, 14));
    data.push_back(rsearch::construct_area_time(0.0, 2.1, 15));
    data.push_back(rsearch::construct_area_time(-0.1, 0.9, 16));
    index->add(data.data(), data.size());
    std::vector<rsearch::query_form> qf_0, qf_1, qf_2;
    qf_0.push_back(rsearch::query_area_time_longtitude_lte(0.7));
    qf_0.push_back(rsearch::query_area_time_latitude_gte(0.7));
    qf_0.push_back(rsearch::query_area_time_timestamp_gte(12));
    std::vector<rsearch::idx_t> quids;
    rsearch::idx_t* uids = NULL;
    int res;
    quids.push_back(0);quids.push_back(2);quids.push_back(4);quids.push_back(6);
    index->query(qf_0.data(), qf_0.size(), &uids, &res);
    std::cout << res << std::endl;
    for (int i = 0; i < res; ++i)
        std::cout << uids[i] << " ";
    std::cout << std::endl;
    free(uids);

    index->query_with_uids(qf_0.data(),qf_0.size(), quids.data(), quids.size(), &uids, &res);
    std::cout << res << std::endl;
    for (int i = 0; i < res; ++i)
        std::cout << uids[i] << " " ;
    std::cout << std::endl;
    free(uids);
    
    qf_1.push_back(rsearch::query_area_time_timestamp_gte(12));
    index->query(qf_1.data(), qf_1.size(), &uids, &res);
    std::cout << res << std::endl;
    for (int i = 0; i < res; ++i)
        std::cout << uids[i] << " " ;
    std::cout << std::endl;
    free(uids);
    delete index;
    return 0;
}

TEST_F(UnitTest, QueryOtherTest) {
    
    //EXPECT_EQ(0, (test_query<float, rsearch::COSINE>(30000, 128, 512, rsearch::DUMMY)) );
    //EXPECT_EQ(0, (test_query<int8_t, rsearch::COSINE>(30000, 128, 512, rsearch::DUMMY)) );
    //EXPECT_EQ(0, (test_query<float, rsearch::COSINE>(30000, 128, 512, rsearch::X86_RAPID)) );
    //EXPECT_EQ(0, (test_query<int8_t, rsearch::COSINE>(30000, 128, 512, rsearch::X86_RAPID)) );
    //EXPECT_EQ(0, (test_query<float, rsearch::EUCLIDEAN>(30000, 128, 512, rsearch::X86_RAPID)) );
    //EXPECT_EQ(0, (test_query<int8_t, rsearch::EUCLIDEAN>(30000, 128, 512, rsearch::X86_RAPID)) );
    //EXPECT_EQ(0, (test_query<float, rsearch::COSINE>(30000, 128, 512, rsearch::X86_PQIVF)) );
    //EXPECT_EQ(0, (test_query<int8_t, rsearch::COSINE>(30000, 128, 512, rsearch::X86_PQIVF)) );
    //EXPECT_EQ(0, (test_query<float, rsearch::EUCLIDEAN>(30000, 128, 512, rsearch::X86_PQIVF)) );
    //EXPECT_EQ(0, (test_query<int8_t, rsearch::EUCLIDEAN>(30000, 128, 512, rsearch::X86_PQIVF)) );
    
    //EXPECT_EQ(0, (test_query<float, rsearch::EUCLIDEAN>(30000, 128, 512, rsearch::FAISS_FLAT)));
    //EXPECT_EQ(0, (test_query<float, rsearch::EUCLIDEAN>(30000, 128, 512, rsearch::FAISS_LSH)));
    //EXPECT_EQ(0, (test_query<float, rsearch::EUCLIDEAN>(30000, 128, 512, rsearch::FAISS_IVFPQ)));
    //EXPECT_EQ(0, (test_query<float, rsearch::EUCLIDEAN>(30000, 128, 512, rsearch::FAISS_HNSW)));


    //EXPECT_EQ(0, (test_query<float, rsearch::COSINE>(1000000, 128, 512, rsearch::X86_PQIVF)) );
    //EXPECT_EQ(0, (test_query<int8_t, rsearch::EUCLIDEAN>(1000000, 128, 512, rsearch::X86_PQIVF)) );
    //EXPECT_EQ(0, (test_query<float, rsearch::EUCLIDEAN>(5000000, 128, 512, rsearch::X86_PQIVF)) );
    //EXPECT_EQ(0, (test_query<int8_t, rsearch::EUCLIDEAN>(5000000, 128, 512, rsearch::X86_RAPID)) );
    //EXPECT_EQ(0, (test_query<int8_t, rsearch::EUCLIDEAN>(5000000, 128, 512, rsearch::X86_RAPID_MULTI_THREAD)) );
    EXPECT_EQ(0, (test_simple_index()) );
    //EXPECT_EQ(0, (test_query<float, rsearch::EUCLIDEAN>(1000000, 128, 512, rsearch::X86_RAPID_MULTI_THREAD)) );
    //EXPECT_EQ(0, (test_query<float, rsearch::EUCLIDEAN>(1000000, 128, 512, rsearch::FAISS_IVFPQ)));
}
