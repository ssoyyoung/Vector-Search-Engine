import os
import cv2
import time
import math

from utils.utils import *
from utils.es_result_parsing import *
from utils.search_algorithm import *
from utils.vector_extractor_v2 import Yolov3
from utils.elasticsearch import Elk
from utils.img_to_vec import imgtovec

from fastapi import FastAPI
from pydantic import BaseModel
from resnet.vector_resnet import Resnet

class Vector(BaseModel):
    img_b64: str
    sex: str = None
    type: str = None
    k: int = 4

app = FastAPI()

YOLO = Yolov3

ELK = Elk
ELK.save_index = 'test_wc13'
ELK.search_index = 'yolo_'
#ELK.search_index = 'yolo_'
ITV = imgtovec


@app.post("/test")
async def search_vec(vector: Vector):
    search_time = time.time()

    img_b64 = vector.img_b64.encode()
    sex = vector.sex
    type = vector.type

    res_list = []
    rb_list = []

    vec = Yolov3.vector_extraction_service(Yolov3, img_b64)
    if len(vec) == 0: return "-1"

    for idx in list(vec.keys()):
        for cnt in range(len(vec[idx])):
            fv = vec[idx][cnt]['feature_vector']
            vector_bs = encode_array(fv.cpu().numpy())
            rb = vec[idx][cnt]['raw_box']
            rb_list.append(rb)

            search_idx = change_index(idx, sex)
            print(search_idx)
            if search_idx == -1: return "-1"

            res = Elk.search_vec(Elk, search_idx, vector_bs, size=4)

            res_list.append(es_parsing(res,rb_list,multi=False))

    print('Interface time for searching vec', time.time()-search_time)
    return res_list


@app.post("/yolo/searchVec_full")
async def search_vector_full(vector: Vector):
    ELK.search_index = 'test_v2_wc13'
    search_time = time.time()
    img_b64 = vector.img_b64.encode()
    type = vector.type

    vec = YOLO.vector_extraction_service_full(YOLO, base_img=img_b64, pooling='max')

    # 검색 이미지에서 물체 검출이 안된경우
    if len(vec) == 0: return "-1"
    if type == 'mac':
        fb = 'feature_vector_mac'
    elif type == 'spoc':
        fb = 'feature_vector_spoc'
    elif type =='cat':
        fb = 'feature_vector_cat'
    vector = vec[fb]
    vector_bs = encode_array(vector)

    search_idx = ELK.search_index
    res = ELK.search_vec_full(ELK, search_idx, vector_bs, type=type)
    es_res = es_parsing(res, multi=False)

    print('Interface time for searching vec', time.time() - search_time)
    return es_res


@app.post("/vector_db/searchVec")
async def search_vector(vector: Vector):
    search_time = time.time()
    #rb_list = []
    es_list = []
    #result_dict = {}

    img_b64 = vector.img_b64.encode()
    gender = vector.sex

    vec = YOLO.vector_extraction_service(YOLO, base_img=img_b64, pooling='max')
    # 검색 이미지에서 물체 검출이 안된경우
    if len(vec) == 0: return "-1"

    box_num = 0
    for idx in list(vec.keys()):
        for count in range(len(vec[idx])):
            vector = vec[idx][count]['feature_vector']
            rb = vec[idx][count]['raw_box']
            vector_bs = encode_array(vector.cpu().numpy())

            search_idx = change_index(idx, gender)
            if search_idx == -1: return "-1"

            res = ELK.search_vec(ELK, search_idx, vector_bs)

            #rb_list.append(rb)
            es_list.append(res)
            #result_dict['obj_'+str(box_num)] = [rb_list,es_list]
            
            box_num += 1

    print('Interface time for searching vec', time.time() - search_time)
    return es_list


@app.post("/yolo/searchVec_box")
async def multi_search_vector(vector: Vector):
    ELK.search_index = 'yolo_'
    search_time = time.time()
    rb_list = []
    vb_list = []
    idx_list = []

    img_b64 = vector.img_b64.encode()
    gender = vector.sex

    vec = YOLO.vector_extraction_service(YOLO, base_img=img_b64, pooling='max')
    # 검색 이미지에서 물체 검출이 안된경우
    if len(vec) == 0: return "-1"
    for idx in list(vec.keys()):
        for count in range(len(vec[idx])):
            fvector = vec[idx][count]['feature_vector']
            rb = vec[idx][count]['raw_box']
            vector_bs = encode_array(fvector.cpu().numpy())

            search_idx = change_index(idx, gender)
            if search_idx == -1: continue
            vb_list.append(vector_bs)
            rb_list.append(rb)
            idx_list.append(search_idx)

    print(idx_list)
    es_time = time.time()
    res = ELK.multi_search_vec(ELK, idx_list, vb_list, vector.k)
    print(res)
    print('search time is...',time.time()-es_time)
    es_res = es_parsing(res, rb_list, multi=True)
    print(es_res)
    print('Interface time for searching vec', time.time() - search_time)
    return es_res

@app.post("/vector_db/imgTovec/")
def save_vector_el():
    sql = "SELECT * FROM `crawling_list` WHERE STATUS=1 AND cat_key = 'WC13'"
    product_list = ITV.connect_db(ITV, sql)
    debug = False

    batch_size = 200
    split_data = 2000

    print('Total count of product list', len(product_list))
    
    save_time = time.time()
    data_dict = {}
    img_path_list = []
    cate_list = []

    for idx, product in enumerate(product_list):

        line = list(product)
        img_path = ITV.base_img_path + line[7] + os.sep + line[8]
        if not os.path.isfile(img_path): continue
        if cv2.imread(img_path) is None: continue
        img_path_list.append(img_path)
        cate_list.append(line[2])
        data_dict[img_path] = [line[0], line[1], line[2], line[3], line[5], line[6], img_path, line[9], line[10], line[15]]
        # 0:id, 1:au_id, 2:cat_key, 3:i_key, 5:img_url, 6:click_url, img_path, 9:group_id, 10:status, 11:gs_bucket
        total_vec = {}
        total_list = []
        if len(img_path_list) % split_data == 0:
            print(data_dict[img_path_list[-1]][0],'...ing')
            n = 0

            for size in range(math.ceil(len(img_path_list) / batch_size)):
                bulk_path = img_path_list[batch_size * n:batch_size * (n + 1)]
                category = cate_list[batch_size * n:batch_size * (n + 1)]
                n += 1

                #vec = YOLO.vector_extraction_batch(YOLO, data=bulk_path, category=category, batch_size=batch_size, pooling='max')
                vec = YOLO.vector_extraction_batch_full(YOLO, data=bulk_path, batch_size=batch_size)

                total_list=total_list+vec

                '''
                for idx in vec.keys():
                    if idx in total_vec.keys():
                        total_vec[idx] = total_vec[idx] + vec[idx]
                    else:
                       total_vec[idx] = vec[idx]
                '''

            if not debug:
                #ELK.bulk_api(ELK, total_vec, data_dict)
                ELK.bulk_api_v2(ELK,total_list, data_dict)
            data_dict = {}
            img_path_list = []
            cate_list = []

    print('Interface time for saving all data', time.time() - save_time)


    return 'send to elk'

@app.post("/vector_db/resnet_oe/")
def resnet():
    sql = "SELECT * FROM `crawling_list` WHERE STATUS=1 LIMIT 10"
    product_list = ITV.connect_db(ITV, sql)
    data_dict = {}
    total_vec = {}
    print('Total count of product list', len(product_list))
    for idx, product in enumerate(product_list):
        line = list(product)
        img_path = ITV.base_img_path + line[7] + os.sep + line[8]
        if not os.path.isfile(img_path): continue
        if cv2.imread(img_path) is None: continue
        data_dict[img_path] = [line[0], line[1], line[2], line[3], line[5], line[6], img_path, line[9], line[10]]

        vec = Resnet.resnet_vector(Resnet, img_path, line[2])

        for idx in vec.keys():
            if idx in total_vec.keys():
                total_vec[idx] = total_vec[idx] + [vec[idx]]
            else:
                total_vec[idx] = [vec[idx]]

    ELK.resnet_bulk_api(ELK, total_vec, data_dict)

    return total_vec


@app.post("/vector_db/resnet/")
def resnet_batch():
    sql = "SELECT * FROM `crawling_list` WHERE STATUS=1 LIMIT 10"
    product_list = ITV.connect_db(ITV, sql)
    debug = True

    batch_size = 10
    split_data = 10

    print('Total count of product list', len(product_list))

    save_time = time.time()
    data_dict = {}
    img_path_list = []
    cate_list = []

    for idx, product in enumerate(product_list):

        line = list(product)
        img_path = ITV.base_img_path + line[7] + os.sep + line[8]
        if not os.path.isfile(img_path): continue
        if cv2.imread(img_path) is None: continue
        img_path_list.append(img_path)
        cate_list.append(line[2])
        data_dict[img_path] = [line[0], line[1], line[2], line[3], line[5], line[6], img_path, line[9], line[10]]
        # 0:id, 1:au_id, 2:cat_key, 3:i_key, 5:img_url, 6:click_url, img_path, 9:group_id, 10:status
        total_vec = {}
        total_list = []
        if len(img_path_list) % split_data == 0:
            print(data_dict[img_path_list[-1]][0], '...ing')
            n = 0

            for size in range(math.ceil(len(img_path_list) / batch_size)):
                bulk_path = img_path_list[batch_size * n:batch_size * (n + 1)]
                category = cate_list[batch_size * n:batch_size * (n + 1)]
                n += 1

                vec = Resnet.resnet_vector_batch(Resnet, data=bulk_path, category=category, batch_size=batch_size, pooling='max')


                for idx in vec.keys():
                    if idx in total_vec.keys():
                        total_vec[idx] = total_vec[idx] + vec[idx]
                    else:
                       total_vec[idx] = vec[idx]


            if not debug:
                # ELK.bulk_api(ELK, total_vec, data_dict)
                ELK.resnet_bulk_api(ELK, total_vec, data_dict)
            data_dict = {}
            img_path_list = []
            cate_list = []

    print('Interface time for saving all data', time.time() - save_time)




