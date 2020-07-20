import rsearch.rsearch as rs
import numpy as np

class handle_rsearch:

    def __init__(self, dimension, topk, method_type, dist_type, var_type):
        self.rsearch = rs.rsearch(dimension, topk, method_type, dist_type, var_type)

    def query(self, x):
        n, d = x.shape
        assert d == self.rsearch.dimension
        sims = None
        if self.rsearch.var_type == rs.FLOAT32:
            sims = np.empty((n, self.rsearch.topk), dtype=np.float32)
        elif self.rsearch.var_type == rs.INT8:
            sims = np.empty((n, self.rsearch.topk), dtype=np.int32)
        uids = np.empty((n, self.rsearch.topk), dtype=np.int32)
        self.rsearch.query(rs.swig_ptr(x), n, rs.swig_ptr(sims), rs.swig_ptr(uids))
        return sims, uids

    def add(self, x):
        n, d = x.shape
        assert d == self.rsearch.dimension
        return self.rsearch.add(rs.swig_ptr(x), n)
        

    def add_with_uids(self, x, uids):
        n, d = x.shape
        assert d == self.rsearch.dimension
        assert uids.shape == (n, ), 'not same nb of vectors as ids'
        return self.rsearch.add_with_uids(rs.swig_ptr(x), rs.swig_ptr(uids), n)

    def change_by_uids(self, x, uids):
        n, d = x.shape
        assert d == self.rsearch.dimension
        assert uids.shape == (n, ), 'not same nb of vectors as ids'
        return self.rsearch.add_with_uids(rs.swig_ptr(x), rs.swig_ptr(uids), n)

    def remove_by_uids(self, uids):
        n = uids.shape
        return self.rsearch.remove_by_uids(self, uids, n)
    
    def query_by_uids(self, uids):
        n, d = uids.shape
        assert d == self.rsearch.dimension
        res = None
        if self.rsearch.var_type == rs.FLOAT32:
            res = np.empty((n, d), dtype=np.float32)
        elif self.rsearch.var_type == rs.INT8:
            res = np.empty((n, d), dtype=np.int8)
        self.rsearch.query(self, rs.swig_ptr(res), n)
        return res
    

    def reset(self):
        return self.rsearch.reset()

    def train(self, x):
        n, d = x.shape
        assert d == self.rsearch.dimension
        return self.rsearch.train(self, x, n)

    def store_data(self, file_name):
        return self.rsearch.store_data(self, file_name)

    def load_data(self, file_name):
        return self.rsearch.load_data(self, file_name)

'''


class handle_simple_index_areatime:
    
    def __init__(self):
        self.rsearch = rs.simple_index_areatime()

    def query(self, x):
        n, d = x.shape
        assert d == self.rsearch.dimension
        idx = rs.IntVector()
        self.rsearch.query(x.data(), x.size(), idx)
        uids = np.empty((idx.size(idx)), dtype=areatime_type)
        uids = 
        return uids

    def 
    def add(self, x):
        n = x.shape
        return self.rsearch.add(rs.swig_ptr(x), n)
        

    def add_with_uids(self, x, uids):
        n = x.shape
        assert uids.shape == (n, ), 'not same nb of vectors as ids'
        return self.rsearch.add_with_uids(rs.swig_ptr(x), rs.swig_ptr(uids), n)

    def change_by_uids(self, x, uids):
        n = x.shape
        assert uids.shape == (n, ), 'not same nb of vectors as ids'
        return self.rsearch.add_with_uids(rs.swig_ptr(x), rs.swig_ptr(uids), n)

    def remove_by_uids(self, uids):
        n = x.shape
        return self.rsearch.remove_by_uids(self, uids, n)

    def query_by_uids(self, *args):
        n = x.shape
        res = np.empty(n, dtype=areatime_type)
        self.rsearch.query(self, rs.swig_ptr(res), n)
        return res

    def reset(self):
        return self.rsearch.reset()

    def store_data(self, file_name):
        return self.rsearch.store_data(self, file_name)

    def load_data(self, file_name):
        return self.rsearch.load_data(self, file_name)

'''
def intpp2array(pp, n):
    p = rs.get_int_p(pp)
    a = np.empty(n, dtype=np.int32)
    rs.memcpy(rs.swig_ptr(a), p, a.nbytes)
    return a

areatime_type = [('longtitude', np.float32), ('latitude', np.float32), ('timestamp', np.int64)]

class handle_simple_index_areatime:
    def __init__(self):
        self.simple_index_areatime = rs.simple_index_areatime()
    
    def add(self, x):
        #n = x.shape
        #return self.simple_index_areatime.add(rs.swig_ptr(x), n)
        return self.simple_index_areatime.add(x.data(), x.size())
    
    def add_with_uids(self, x, uids):
        assert x.size() == uids.size()
        return self.simple_index_areatime.add_with_uids(x, uids.data(), x.size())
    
    def change_by_uids(self, x, uids):
        assert x.size() == uids.size()
        return self.simple_index_areatime.change_by_uids(x, uids.data(), x.size())

    def remove_by_uids(self, uids):
        return self.simple_index_areatime.remove_by_uids(uids.data(), uids.size())

    def query_by_uids(self, uids):
        
        n = uids.size()
        x = AreaTimeVector()
        x.resize(n)
        self.simple_index_areatime.query_by_uids(uids.data(), uids.size(), x.data())
        return x

    def reset(self):
        return self.simple_index_areatime.reset()

    def store_data(self, file_name):
        return self.simple_index_areatime.store_data(file_name)

    def load_data(self, file_name):
        return self.simple_index_areatime.load_data(file_name)

    def query(self, x):
        idx = rs.get_int_pp()
        res = rs.get_int_p()
        self.simple_index_areatime.query(x.data(), x.size(), idx, res)
        n = rs.get_int(res)
        a = intpp2array(idx, n)
        return a

    def query_with_uids(self, x, uids):
        idx = rs.get_int_pp()
        res = rs.get_int_p()
        self.simple_index_areatime.query_with_uids(x.data(), x.size(), uids.data(), uids.size(), idx, res)
        n = rs.get_int(res)
        a = intpp2array(idx, n)
        return a

def vector2array(v):
    classname = v.__class__.__name__
    assert classname.endswith('Vector')
    dtype = np.dtype(vector_name_map[classname[:-6]])
    a = np.empty(v.size(), dtype=dtype)
    if v.size() > 0:
        rs.memcpy(swig_ptr(a), v.data(), a.nbytes)
    return a
