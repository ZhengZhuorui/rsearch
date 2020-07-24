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
import traceback

dimension = 128
print('target1')
glb = rs_util.utils()
print('target2')
# def select_dataset(request):

def select_dataset(request):
    print(request.POST)
    id = request.POST['datasetID']
    glb.load_dataset(id)
    '''
    TODO: 切换sqlite数据库
    '''
    return JsonResponse({'result':0})

def query_dataset(request):
    print('[query dataset]')
    res = Dataset.objects.all()
    res_lt = list(map(lambda x: {'name':x.name,'id':x.id}, res))
    print(res_lt)
    return JsonResponse({'result':0, 'dataset':res_lt})


def insert_dataset(request):
    print(request.POST['datasetName'])
    name = request.POST['datasetName']
    file_name = str(int(time.time())) + '.sqlite3'
    if 'data' in request.FILES:
        file_name = request.FILES['data'].name
    dataset_path = os.path.join(rs_util.dataset_dir, file_name)
    glb.insert_dataset(name, dataset_path)
    return JsonResponse({'result':0})
    
def insert_data(request):
    print(request.POST)
    #time = request.POST.get('time')
    image = request.FILES['image']
    image_name = request.FILES['image'].name
    image_type = os.path.splitext(image_name)[1].lower()
    _time = request.POST['time']
    longtitude = request.POST['longtitude']
    latitude = request.POST['latitude']
    text = request.POST['text']
    timestamp = int(time.time())

    if _time != '':
        timeArray = time.strptime(_time, "%Y-%m-%d %H:%M:%S")
        timestamp=int(time.mktime(timeArray))
   
    glb.insert_data(timestamp, longtitude, latitude, text, image, image_type)
    print('insert_data end.')
    # result_id = glb.insert_data(timeStamp, longtitude_v, latitude_v, feature, image, image_type)
    # return JsonReponse({'result':0, 'resultID':result_id})
    return JsonResponse({'result':0})
    

def remove_data(request):
    ID = int(request.POST['ID'])
    print('remove_data', ID)
    try:
        glb.remove_data(ID)       
        return JsonResponse({'result':0})
    except Exception as e:
        print(e)
        print(traceback.format_exc())
        return JsonResponse({'result':1})

def query(request):
    startTime = request.POST['startTime']
    endTime = request.POST['endTime']
    longtitude_lte = request.POST['longtitude_lte']
    longtitude_gte = request.POST['longtitude_gte']
    latitude_lte = request.POST['latitude_lte']
    latitude_gte = request.POST['latitude_gte']
    text = None
    if request.POST['text'] != '':
        text = request.POST['text']
    image = None
    image_name = ''
    image_type = ''
    image_path = ''
    if 'image' in request.FILES:
        image = request.FILES['image']
        image_name = request.FILES['image'].name
        image_type = os.path.splitext(image_name)[1].lower()
        image_path = glb.save_image(image, image_type)
    lt = rs.QueryFormVector()
    print('[query] time:')
    print(time.time())
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(int(time.time()))))
    if startTime != '':
        print(startTime)
        timeArray = time.strptime(startTime, "%Y-%m-%d %H:%M:%S")
        timeStamp = int(time.mktime(timeArray))
        print(timeStamp)
        queryForm = rs.query_area_time_timestamp_gte(timeStamp)
        lt.push_back(queryForm)
    if endTime != '':
        timeArray = time.strptime(endTime, "%Y-%m-%d %H:%M:%S")
        timeStamp = int(time.mktime(timeArray))
        print(timeStamp)
        queryForm = rs.query_area_time_timestamp_lte(timeStamp)
        lt.push_back(queryForm)

    if longtitude_lte != '':
        s = glb.degree_translate(longtitude_lte)
        queryForm = rs.query_area_time_longtitude_lte(s)
        lt.push_back(queryForm)

    if longtitude_gte != '':
        s = glb.degree_translate(longtitude_gte)
        queryForm = rs.query_area_time_longtitude_gte(s)
        lt.push_back(queryForm)

    if latitude_lte != '':
        s = glb.degree_translate(latitude_lte)
        queryForm = rs.query_area_time_latitude_lte(s)
        lt.push_back(queryForm)

    if latitude_gte != '':
        s = glb.degree_translate(latitude_gte)
        queryForm = rs.query_area_time_latitude_gte(s)
        lt.push_back(queryForm)
    
    feature = np.empty((1, dimension), dtype=np.float32)

    if text != None and image == None:
        feature = glb.textEncoding(text)
    if text == None and image != None:
        feature = glb.imageEncoding(image_path)
    if text != None and image != None:
        feature = glb.imagetextEncoding(text, image_path)
    
    feature = feature[np.newaxis, :]
    sims = np.zeros((1, 128), dtype=np.float32)
    res = np.zeros((1, 128), dtype=np.int32)

    if text != None or image != None:
        print('t1')
        sims, res = glb.probe.query(feature)
    print(res) 
    res = np.squeeze(res)
    res_lt = []
    if text != None or image != None:
        print('t2')
        res = glb.simple_index.query_with_uids(lt, res)
    else:
        print('t3')
        res = glb.simple_index.query(lt)
    
    print(res)
    id_lt = res.tolist()
    vec = glb.simple_index.query_by_uids(res)
    for i in range(vec.size()):
        if id_lt[i] != -1:
            res_lt.append({'lng':vec.at(i).longtitude, 'la':vec.at(i).latitude, 't':vec.at(i).timestamp, 'id':id_lt[i]})

    # result example: 
    # res_lt = [{'lng':0, 'la':0, 't':0, 'id':0}]

    result = {'result':0, 'query_result':res_lt}
    print(result)
    return JsonResponse(result)

def get_image(request):
    image_id = int(request.GET['id'])
    try:
        '''
        TODO: 操作sqlite读写文件，ORM模式如下：
        data = RemoteSensing.get(id=image_id)
        image_path = data.file_name


        下面只返回一张样例图片
        '''
        print('get_image', image_id)
        image_path = glb.get_image(image_id)
        print('get_image', image_path)
        image_data = None
        #image_path = os.path.join(rs_util.image_dir, 'example.png')
        with open(image_path, 'rb') as f:
            image_data = f.read()
        return HttpResponse(image_data, content_type='image/png')
    except Exception:
        pass


