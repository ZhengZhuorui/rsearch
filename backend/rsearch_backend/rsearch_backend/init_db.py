import rsearch.rsearch as rs
import rsearch.handle_rsearch as hrs
import rsearch_backend.database as database
import rsearch_backend.utils as utils
import ast
from django.http import *
import shutil
import os
import numpy as np
import time


db_path = '/home/mdm/Image1000.db'
db_image_path = '/home/mdm/Images1000/'
image_dir = './image/'
dimension = 128
def degree_translate(degree_s):
    a = degree_s.split('.')
    c = a[2][:-1]
    f = a[2][-1]
    degree = int(a[0]) + int(a[1]) / 60 + int(c) / 3600
    if f == 'S' or f == 'W':
        degree = -degree
    return degree

def get_timestamp(s):
    timeArray = time.strptime(s, '%Y-%m-%d')
    return int(time.mktime(timeArray))

def open_image(path):
    f = open(path, 'rb')
    image = f.read()
    f.close()
    return image

def init_db(request):
    db = database.DBConnector(db_path)
    res = db.select()
    print(res[0])
    print(len(res))
    glb = utils.utils()
    glb.insert_dataset('Image1000', './dataset/Image1000.sqlite3')
    # glb.select_dataset('Image1000')
    data_vec = rs.AreaTimeVector()
    
    feature = np.empty((len(res), dimension), dtype=np.float32)
    for i,x in enumerate(res):        
        lng = degree_translate(x[3])
        lat = degree_translate(x[2])
        print(x[3], lng, x[2], lat)
        t = get_timestamp(x[1])
        data_vec.push_back(rs.construct_area_time(lng, lat, t))
        print(type(x[4]))
        ft = ast.literal_eval(x[4])
        feature[i] = np.array(ft, dtype=np.float32)
        image_path = os.path.join(db_image_path, x[5])
        image = open_image(image_path)
        image_path = os.path.join(image_dir, x[5])
        
        f = open(image_path, 'wb')
        f.write(image)
        f.close()
        
        glb.sqliteDB.insert(None, str(t), str(lat), str(lng), ft, image_path)
    print('t3')

    glb.probe.add(feature)
    glb.probe.store_data(glb.dataset.path1)

    glb.simple_index.add(data_vec)
    glb.simple_index.store_data(glb.dataset.path2)

    return JsonResponse({'result':0})
    
