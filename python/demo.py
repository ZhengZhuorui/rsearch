import rsearch
import numpy as np

if __name__ == '__main__':
    probe = rsearch.create_probe_float(512, 128, rsearch.rsearch_X86_RAPID, rsearch.rsearch_EUCLIDEAN)
    ga = None
    probe.create_gallery(ga)
    data = np.random.rand(1000, 512)

    test_data = data[10:20,:]
    ga.add(data, 1000)
    uids=np.random.zeros(10*128, dtype=np.int32)
    sims=np.random.zeros(10*128, dtype=np.float32)
    probe.query(test_data, 10, ga, uids, sims)
    print(uids)
    print(sims)
