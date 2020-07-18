from data.models import Dataset
from django.http import *
import rsearch.handle_rsearch as hrs
import rsearch.rsearch as rs
import time

probe = hrs.handle_rsearch(512 , 128, rs.X86_RAPID, rs.EUCLIDEAN)
probe.init()
'''
Or TODO:
probe.add()
'''
simple_index = hrs.handle_simple_index()

glb_dataset_id = ""

def degree_translate(s):
    ds = longtitude_lt.split('D')
    d = int(ds[0])
    fs = ds[1].split('F')
    f = int(fs[0])
    m = int (fs[1])
    return d * 3600 + f * 60 + m

def select_dataset(request):
    dataset_id = request.POST['datasetID']
    if glb_dataset_name != dataset_name:
        dataset = Dataset.get(name = Dataste.name)
        probe.load_data(dataset.path1)
        '''
            Or TODO:
            data = get_data(dataset_name)
            probe.add(data)
        '''

        simple_index.reset()
        simple_index.load(dataset.path2)

def add(request):
    
    time = request.POST['time']
    longtitude = request.POST['longtitude']
    latitude = request.POST['latitude']
    text = request.POST['text']
    '''
    TODO:
    vector = np.empty((1, dimension), dtype=np.float32)
    if text != None and image = None:
        vector = model1(text)
    if text = None and image != None:
        vector = model2(image)
    if text != None and image != None:
        vector = model3(text, image)
    _add(vector)
    '''


def remove(request):
    ID = int(request.POST['ID'])
    vec = IntVector()
    vec.push_back(ID)
    simple_index.remove_by_uids(vec)
    probe.remove_by_uids(vec)

def query(request):
    startTime = request.POST['startTime']
    endTime = request.POST['endTime']
    longtitude_lte = request.POST['longtitude_lt']
    longtitude_gte = request.POST['longtitude_gt']
    latitude_lte = request.POST['latitude_lt']
    latitude_gte = request.POST['latitude_gt']
    text = None
    if 'text' in request.POST:
        text = request.POST['text']
    image = None
    if 'image' in request.FILES:
        image = request.FILES['image']
    have_timestamp = True


    lt = rsearch.QueryFormVector()
    
    if startTime != '':
        timeArray = time.strptime(startTime, "%Y-%m-%d %H:%M:%S")
        timeStamp = int(time.mktime(timeArray))
        queryForm = rsearch.set_timestamp_gte(timeStamp)
        lt.append(queryForm)
    if endTime != '':
        timeArray = time.strptime(endTime, "%Y-%m-%d %H:%M:%S")
        timeStamp = int(time.mktime(timeArray))
        queryForm = rsearch.set_timestamp_lte(timeStamp)
        lt.append(queryForm)

    if longtitude_lte != '':
        queryForm = rsearch.set_longtitude_lte(degree_translate(longtitude_lte))
        lt.append(queryForm)

    if longtitude_gte != '':
        queryForm = rsearch.set_longtitude_gte(degree_translate(longtitude_gte))
        lt.append(queryForm)

    if latitude_lte != '':
        queryForm = rsearch.set_latitude_lte(degree_translate(latitude_lte))
        lt.append(queryForm)

    if latitude_gte != '':
        queryForm = rsearch.set_latitude_gte(degree_translate(latitude_gte))
        lt.append(queryForm)

    
    '''
    TODO:
    vector = np.empty((1, dimension), dtype=np.float32)
    if text != None and image = None:
        vector = model1(text)
    if text = None and image != None:
        vector = model2(image)
    if text != None and image != None:
        vector = model3(text, image)
    '''

    sims = np.zeros((1, 128), dtype=np.float32)
    uids = np.zeros((1, 128), dtype=np.int32)

    if text != None or image != None:
        sims, uids = probe.query(vector)
    
    if lt.size() != 0:
        res = simple_index.query(lt.data(), lt.size(), uids)
    # print(res)
    '''
    TODO:
    image = find(res)
    return image
    '''
    return HttpResponse()
    

    
    
    

