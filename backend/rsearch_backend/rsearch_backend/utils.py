from model.models import Dataset
from model.models import RemoteSensing
from django.http import *
import rsearch.handle_rsearch as hrs
import rsearch.rsearch as rs
import time
import random
import numpy as np
import os

import rsearch_backend.Encoder as Encoder
from rsearch_backend.database import DBConnector

#the image directory
image_dir = "./image/"

# the dataset directory
dataset_dir = "./dataset/"

#feature dimension
dimension = 128
topk = 128
method_type = rs.X86_RAPID
dist_type = rs.EUCLIDEAN
var_type = rs.FLOAT32

class utils:
    probe = hrs.handle_rsearch(dimension, 128, method_type, dist_type, var_type)
    simple_index = hrs.handle_simple_index_areatime()
    def __init__(self):
        self.dataset = None
        self.encoder = Encoder.Encoder()
        self.sqliteDB = None
        #self.probe.init()

    def textEncoding(self, text):
        '''
        TODO: 从文本提取特征

        '''
        return self.encoder.textEncoding(text)
        return np.empty((1, dimension), dtype=np.float32)

    def imageEncoding(self, imagepath):
        '''
        TODO: 从图像提取特征

        '''
        return self.encoder.imageEncoding(imagepath)
        return np.empty((1, dimension), dtype=np.float32)
    def imagetextEncoding(self, text, imagepath):
        '''
        TODO: 从图像和文本提取特征

        '''
        txtvect = self.encoder.textEncoding(text)
        imgvect = self.encoder.imageEncoding(image)
        return (txtvect + imgvect) / 2
        return np.empty((1, dimension), dtype=np.float32)
    
    def get_image(self, id):
        return self.sqliteDB.select_id(id)

    def save_image(self, image, image_type):
        file_name = str(int(time.time())) + image_type
        file_name = os.path.join(image_dir, file_name)
        with open(file_name, 'wb') as f:
            for content in image.chunks():
                f.write(content)
        return file_name

    def save_image_with_filename(self, image, file_name):
        with open(file_name, 'wb') as f:
            for content in image.chunks():
                f.write(content)
        return

    def insert_data(self, time, longtitude, latitude, feature, image, image_type):
        file_name = self.save_image(image, image_type)
        self.sqliteDB.insert(NULL, time, latitude, longtitude, feature, file_name)
        '''
            TODO: 从sqlite中增加数据，ORM模式如下：
            feature_text = array2text(feature)
            data = RemoteSensing(time, longtitude, latitude, feature_text, file_name)
            data.save()
        '''
        self.probe.add(feature)
        self.simple_index.add(rs.construct_area_time(longtitude, latitude, time))
        self.probe.store_data(self.dataset.path1)
        self.simple_index.store_data(self.dataset.path2)

    def remove_data(self, id):

        '''
            TODO: 从sqlite中删除数据，ORM模式如下：
            data = RemoteSensing.objects.get(id=id)
            data.remove()
        '''
        self.sqliteDB.delete(id)
        ID_array = np.array([id])
        self.probe.remove_by_uids(ID_array)
        self.probe.remove_by_uids(ID_array)
    

    def load_dataset(self, id):
        try:
            self.dataset = Dataset.objects.get(id=id)
            self.sqliteDB = DBConnector(self.dataset.database_path)
            self.probe.load_data(self.dataset.path1)
            self.simple_index.load_data(self.dataset.path2)
        except Exception:
            pass

    def store_dataset(self):
        self.probe.store_data(self.dataset.path1)
        self.simple_index.store_data(self.dataset.path2)
    
    def insert_dataset(self, name, dataset_path):
        path1 = str(int(time.time())) + '.d1'
        path1 = os.path.join(image_dir, path1)
        path2 = str(int(time.time())) + '.d1'
        path2 = os.path.join(image_dir, path2)
        print(path1, path2)
        f = open(path1, 'wb')
        f.close()
        f = open(path2, 'wb')
        f.close()
        self.dataset = Dataset(name, dataset_path, path1, path2)
        self.sqliteDB = DBConnector(dataset_path)
        self.dataset.save()


    def degree_translate(self, s):
        ds = s.split(' ')
        d = float(ds[0])
        if ds[1][0] == 'W' or ds[1][0] == 'S':
            d = -d
        return d

