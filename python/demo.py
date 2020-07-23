import rsearch.handle_rsearch as h_rs
import rsearch.rsearch as rs
import numpy as np

if __name__ == '__main__':
    probe = h_rs.handle_rsearch(512, 128, rs.X86_RAPID, rs.EUCLIDEAN, rs.FLOAT32)
    print('target 1')
    data = np.random.rand(1000, 512)
    data = data.astype(np.float32)
    print(data.dtype)
    test_data = data[10:20,:]
    probe.add(data)
    print('target2')
    uids, sims = probe.query(test_data)
    print('target3')
    print(uids)
    print(sims)

    index = h_rs.handle_simple_index_areatime()
    
    #data_1 = np.array([(0.5, 1.5, 10),(-0.5, 0.5, 11),(0.8, 1.0, 12),(0.2, 0.6, 13),(0.6, -2.0, 14),(0.0, 2.1, 15),(-0.1, 0.9, 16)], dtype=h_rs.areatime_type)
    
    data_vec = rs.AreaTimeVector()
    data_vec.push_back(rs.construct_area_time(0.5, 1.5, 10))
    data_vec.push_back(rs.construct_area_time(-0.5, 0.5, 11))
    data_vec.push_back(rs.construct_area_time(0.8, 1.0, 12))
    data_vec.push_back(rs.construct_area_time(0.2, 0.6, 13))
    data_vec.push_back(rs.construct_area_time(0.6, -2.0, 14))
    data_vec.push_back(rs.construct_area_time(0.0, 2.1, 15))
    data_vec.push_back(rs.construct_area_time(-0.1, 0.9, 16))
    
    index.add(data_vec)

    qf_0 = rs.QueryFormVector()
    qf_1 = rs.QueryFormVector()
    qf_0.push_back(rs.query_area_time_longtitude_lte(0.7))
    qf_0.push_back(rs.query_area_time_latitude_gte(0.7))
    qf_0.push_back(rs.query_area_time_timestamp_gte(12))
    qf_1.push_back(rs.query_area_time_timestamp_gte(12))
    quids = rs.IntVector()
    quids.push_back(0)
    quids.push_back(2)
    quids.push_back(4)
    quids.push_back(6)    
    uids = index.query(qf_0)
    print(uids)
    uids = index.query_with_uids(qf_0, quids);
    print(uids)
    uids = index.query(qf_1)
    print(uids)

    index.store_data('example.d2')
    index.reset()
    index.load_data('example.d2')
    index.query(qf_0)

