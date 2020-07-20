from model.models import Dataset
from model.models import RemoteSensing
from django.http import *
import rsearch.handle_rsearch as hrs
import rsearch.rsearch as rs
import time
import random
import rsearch_backend.utils as rs_util
import numpy as np
import os

dimension = 128
print('target1')
glb = rs_util.utils()
print('target2')
# def select_dataset(request):

def select_dataset(request):
    id = request.POST['datasetID']
    glb.load_dataset(id)
    '''
    TODO: 切换sqlite数据库
    '''
    return HttpResponse({'result':0})

def query_dataset(request):
    res = Dataset.objects.all()
    print(res)
    return HttpResponse(res)


def insert_dataset(request):
    print(request.POST['datasetName'])
    name = request.POST['datasetName']
    file_name = str(int(time.time())) + '.sqlite3'
    if 'data' in request.FILES:
        file_name = request.FILES['data'].name
    dataset_path = os.path.join(rs_util.dataset_dir, file_name)
    glb.insert_dataset(name, dataset_path)
    return HttpResponse({'result':0})
    
def insert_data(request):
    
    time = request.POST['time']
    longtitude = request.POST['longtitude']
    latitude = request.POST['latitude']
    text = request.POST['text']
    image = request.FILES['image']
    image_name = request.FILES['image'].name
    image_type = os.path.splitext(image_name)[1].lower()

    if text != None and image == None:
        feature = glb.textEncoding(text)
    if text == None and image != None:
        feature = glb.imageEncoding(image)
    if text != None and image != None:
        feature = glb.imagetextEncoding(image)

    glb.insert_data(time, longtitude, latitude, feature, image, image_type)
    

def remove_data(request):
    ID = int(request.POST['ID'])
    try:
        glb.remove_data(ID)        
        return HttpResponse({'result':0})
    except Exception:
        return HttpResponse({'result':1})

def query(request):
    startTime = request.POST['startTime']
    endTime = request.POST['endTime']
    longtitude_lte = request.POST['longtitude_lte']
    longtitude_gte = request.POST['longtitude_gte']
    latitude_lte = request.POST['latitude_lte']
    latitude_gte = request.POST['latitude_gte']
    text = None
    if 'text' in request.POST:
        text = request.POST['text']
    image = None
    if 'image' in request.FILES:
        image = request.FILES['image']
    image_name = request.FILES['image'].name
    image_type = os.path.splitext(image_name)[1].lower()
    
    file_name = glb.save_image(image, image_type)
    lt = rs.QueryFormVector()
    
    if startTime != '':
        timeArray = time.strptime(startTime, "%Y-%m-%d %H:%M:%S")
        timeStamp = int(time.mktime(timeArray))
        queryForm = rs.query_area_time_timestamp_gte(timeStamp)
        lt.append(queryForm)
    if endTime != '':
        timeArray = time.strptime(endTime, "%Y-%m-%d %H:%M:%S")
        timeStamp = int(time.mktime(timeArray))
        queryForm = rs.query_area_time_timestamp_lte(timeStamp)
        lt.append(queryForm)

    if longtitude_lte != '':
        s = glb.degree_translate(longtitude_lte)
        queryForm = rs.query_area_time_longtitude_lte(s)
        lt.append(queryForm)

    if longtitude_gte != '':
        glb.degree_translate(longtitude_gte)
        queryForm = rs.query_area_time_longtitude_gte(s)
        lt.append(queryForm)

    if latitude_lte != '':
        glb.degree_translate(latitude_lte)
        queryForm = rs.query_area_time_latitude_lte(s)
        lt.append(queryForm)

    if latitude_gte != '':
        glb.degree_translate(latitude_gte)
        queryForm = rs.query_area_time_latitude_gte(s)
        lt.append(queryForm)
    
    feature = np.empty((1, dimension), dtype=np.float32)

    if text != None and image == None:
        feature = glb.textEncoding(text)
    if text == None and image != None:
        feature = glb.imageEncoding(file_name)
    if text != None and image != None:
        feature = glb.imagetextEncoding(image)
    

    sims = np.zeros((1, 128), dtype=np.float32)
    uids = np.zeros((1, 128), dtype=np.int32)

    if text != None or image != None:
        sims, uids = glb.probe.query(lt)
    
    res = None
    res_lt = []
    if lt.size() != 0:
        res = glb.simple_index.query_with_uids(lt, uids)
        id_lt = res.tolist()
        vec = glb.simple_index.query_by_uids(res)
        for i in range(vec.size()):
            res_lt.append({'lng':vec.at(i).longtitude, 'la':vec.at(i).latitude, 't':vec.at(i).timestamp, 'id':id_lt[i]})

    # result example: 
    res_lt = [{'lng':0, 'la':0, 't':0, 'id':0}]

    result = {'result':0, 'query_result':res_lt}
    return HttpResponse(result)

def get_image(request):
    image_id =  request.GET['id']
    try:
        '''
        TODO: 操作sqlite读写文件，ORM模式如下：
        data = RemoteSensing.get(id=image_id)
        image_path = data.file_name


        下面只返回一张样例图片
        '''
        image_path = os.path.join(rs_util.image_dir, 'example.png')
        with open(image_path, 'rb') as f:
            image_data = f.read()
        return HttpResponse(image_data, content_type='image/png')
    except Exception:
        pass


