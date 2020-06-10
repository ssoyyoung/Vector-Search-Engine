from fastapi import APIRouter
from model import model
import os
import cv2
import time
import math
import glob

from utils.utils import *
from utils.es_result_parsing import *
from utils.search_algorithm import *
from utils.elasticsearch import Elk
from utils.img_to_vec import imgtovec
from utils.checkDouble import Image2Vector

import base64
from io import BytesIO
from PIL import Image
from fastapi import FastAPI
from pydantic import BaseModel
from utils.vector_extractor_v2 import Yolov3

# APIRouter() 객체 생성
router = APIRouter()

# 상위 prefix로 설정 한 search_vec/box로 연결.
@router.post("/box")
async def search_vec(vector: model.Vector):
    es_res = {}
    search_time = time.time()
    img_b64, sex, type, model = vector.img_b64.encode(), vector.sex, vector.type, vector.model
    rb_list, vb_list, idx_list, cate_list = [], [], [], []

    if model == 'yolo':
        vec = Yolov3.vector_extractor_by_model_service(Yolov3, img_b64, vector.type)
        if type == 'mac': vector_type = 'yolo_vector_mac'
        elif type == 'spoc': vector_type = 'yolo_vector_spoc'
        elif type == 'cat': vector_type = 'yolo_vector_cat'
    elif 'resnet' in model:
        vec = Yolov3.vector_extractor_by_model_service(Yolov3, img_b64, vector.model)
        vector_type = 'resnet_vector'
    else:
        return "-1"

    if len(vec) == 0: return "-1"

    d = time.time()
    for idx in list(vec.keys()):
        for cnt in range(len(vec[idx])):
            vector_bs = encode_array(vec[idx][cnt]['feature_vector'])
            rb = vec[idx][cnt]['raw_box']
            cate = vec[idx][cnt]['category']

            search_idx = change_index(idx, sex)
            if search_idx == -1: continue

            #box_size = (rb[2]-rb[0])*(rb[3]-rb[1])
            # if box_size < 0.07: czzzontinue

            """ cx, cy = (rb[2]-rb[0])/2+rb[0], (rb[3]-rb[1])/2+rb[1]
            if not 0.1 < cx < 0.9 or not 0.1 < cy < 0.9:
                print('cx or cy out of bound', (cx, cy))
                continue """

            rb_list.append(rb)
            vb_list.append(vector_bs)
            idx_list.append(search_idx)
            cate_list.append(cate)
    
    print("data collect", time.time()-d)

    if "1024" in model:
        el_idx = "resnet1024"
    elif "512" in model:
        el_idx = "pirs"
    else:
        el_idx=""
    
    idx_list = [el_idx+idx for idx in idx_list]
    print(idx_list)

    if len(rb_list) == 0:
        return "-1"
    
    el_time = time.time()
    res = Elk.multi_search_vec(Elk, idx_list, vb_list, vector.k, vector_type)
    
    print("multi search ", time.time()-el_time)

    parsing_time = time.time()
    es_res['es_res'] = es_parsing(res, rb_list, cate_list, multi=True)
    print("parsing time", time.time()-parsing_time)
    
    es_res['service_time'] = time.time() - search_time
    print('Interface time for searching vec', time.time()-search_time)

    return es_res

img2vec = Image2Vector(cuda=True)



@router.post("/exist")
async def search_exist(d_img: model.DoubleIMG):
    total_time = time.time()
    save_list = []
    b64_img, c_key, img_url = d_img.img_b64, d_img.c_key, d_img.img_url
    # base64 to PIL image....
    img = Image.open(BytesIO(base64.b64decode(b64_img))).convert('RGB')

    vec = img2vec.get_vector(img)
    res = Elk.search_vec(Elk, "search_img", encode_array(vec))

    # initialize
    status = False
    if json.loads(res)['hits']['hits']==[]:
        Elk.save_vector_ps(Elk, vec, c_key, img_url)
        return status

    # image check process
    try:
        max_score = json.loads(res)['hits']['hits'][0]['_score']
        if max_score> 0.98:
            status = True
            return status
        else:
            status = False
            return status
    finally:
        if status is False:
            Elk.save_vector_ps(Elk, vec, c_key, img_url)

    return "DONE"

