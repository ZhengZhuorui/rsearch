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
import traceback

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
        return self.encoder.textEncoding(text)
        return np.empty((1, dimension), dtype=np.float32)

    def imageEncoding(self, imagepath):
        return self.encoder.imageEncoding(imagepath)
        return np.empty((1, dimension), dtype=np.float32)
    def imagetextEncoding(self, text, imagepath):
        txtvect = self.encoder.textEncoding(text)
        imgvect = self.encoder.imageEncoding(imagepath)
        return (txtvect + imgvect) / 2
        return np.empty((1, dimension), dtype=np.float32)
    
    def get_image(self, id):
        image_path = self.sqliteDB.select_id(id)
        print(image_path)
        return image_path

    def save_image(self, image, image_type):
        file_name = str(time.time()) + image_type
        file_name = os.path.join(image_dir, file_name)
        with open(file_name, 'wb') as f:
            for content in image.chunks():
                f.write(content)
        return file_name

    def save_image_with_name(self, image, file_name):
        path = os.path.join(image_dir, file_name)
        print('[save_image_with_path]',path)
        with open(path, 'wb') as f:
            for content in image.chunks():
                f.write(content)
        return path

    def insert_data(self, time, longtitude_s, latitude_s, text, image, image_type):
        print(time, longtitude_s, latitude_s, text)
        longtitude = 0.0
        latitude = 0.0
        image_path = self.save_image(image, image_type)
        feature = np.empty((1,dimension), dtype=np.float32)
        if longtitude_s != '':
            longtitude = self.degree_translate(longtitude_s)
        if latitude_s != '':
            latitude = self.degree_translate(latitude_s)
        if text != None and image == None:
            feature = self.textEncoding(text)
        if text == None and image != None:
            feature = self.imageEncoding(image_path)
        if text != None and image != None:
            feature = self.imagetextEncoding(text, image_path)
        feature = feature[np.newaxis, :]
        print('insert data t1')
        print(feature)
        print(feature.shape)
        feature_s = str(list(feature.tolist()))
        self.sqliteDB.insert(None, time, latitude, longtitude, feature_s, image_path)
        print('insert data t2')
        vec = rs.AreaTimeVector()
        vec.push_back(rs.construct_area_time(longtitude, latitude, time))
        print('insert data t2.1')
        self.probe.add(feature)
        print('insert data t2.2')
        self.simple_index.add(vec)
        print('insert data t3')
        self.probe.store_data(self.dataset.path1)
        self.simple_index.store_data(self.dataset.path2)
        print('insert data t4')

    def remove_data(self, id):
        print('t1')
        self.sqliteDB.delete(id+1)
        print('t2')
        ID_array = np.array([id], dtype=np.int32)
        print('t3', ID_array)
        self.probe.remove_by_uids(ID_array)
        print('t4')
        self.simple_index.remove_by_uids(ID_array)
        print('t5')
        self.probe.store_data(self.dataset.path1)
        print('t6')
        self.simple_index.store_data(self.dataset.path2)
        print('t7')


    def load_dataset(self, _id):
        print('[load_dataset]')
        if self.dataset!= None and _id == self.dataset.id:
            return
        try:
            print(_id)
            self.dataset = Dataset.objects.get(id=_id)
            print(self.dataset.database_path)
            self.sqliteDB = DBConnector(self.dataset.database_path)
            self.probe.reset()
            self.simple_index.reset()
            self.probe.load_data(self.dataset.path1)
            self.simple_index.load_data(self.dataset.path2)
        except Exception as e:
            print(e)
            print(traceback.format_exc())
            

    def store_dataset(self):
        self.probe.store_data(self.dataset.path1)
        self.simple_index.store_data(self.dataset.path2)
    
    def insert_dataset(self, name, dataset_path):
        path1 = str(int(time.time())) + '.d1'
        path1 = os.path.join(dataset_dir, path1)
        path2 = str(int(time.time())) + '.d2'
        path2 = os.path.join(dataset_dir, path2)
        print(path1, path2)
        self.probe.store_data(path1)
        self.simple_index.store_data(path2)
        self.dataset = Dataset(name=name, database_path=dataset_path, path1=path1, path2=path2)
        self.sqliteDB = DBConnector(dataset_path)
        self.dataset.save()


    def degree_translate(self, s):
        ds = s.split(' ')
        d = float(ds[0])
        if ds[1][0] == 'W' or ds[1][0] == 'S':
            d = -d
        return d

