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

from io import BytesIO
from PIL import Image
from fastapi import FastAPI
from pydantic import BaseModel
from utils.vector_extractor_v2 import Yolov3

app = FastAPI()
Elk.save_index = 'pirs_'
Elk.search_index = 'pirs_'

# input data type
class Vector(BaseModel):
    img_b64: str #이미지
    sex: str = None #성별
    type: str = 'mac' #벡터추출방식 (mac, spoc, cat, none)
    k: int = 4 #k갯수
    model: str = 'yolo' #벡터추출모델
    state: str = 'true' #초기 상태값 STATUS

# input data type
class DoubleIMG(BaseModel):
    img_b64: str #이미지
    c_key: str
    img_url: str



@app.post("/search_vec/box")
async def search_vec(vector: Vector):
    es_res = {}
    search_time = time.time()
    img_b64, sex, type, model = vector.img_b64.encode(), vector.sex, vector.type, vector.model
    rb_list, vb_list, idx_list, cate_list = [], [], [], []

    if model == 'yolo':
        vec = Yolov3.vector_extractor_by_model_service(Yolov3, img_b64, vector.type)
        if type == 'mac': vector_type = 'yolo_vector_mac'
        elif type == 'spoc': vector_type = 'yolo_vector_spoc'
        elif type == 'cat': vector_type = 'yolo_vector_cat'
    elif model == 'resnet':
        vec = Yolov3.vector_extractor_by_model_service(Yolov3, img_b64, vector.model)
        vector_type = 'resnet_vector'
    else:
        return "-1"

    if len(vec) == 0: return "-1"

    for idx in list(vec.keys()):
        for cnt in range(len(vec[idx])):
            vector_bs = encode_array(vec[idx][cnt]['feature_vector'])
            rb = vec[idx][cnt]['raw_box']
            cate = vec[idx][cnt]['category']

            search_idx = change_index(idx, sex)
            if search_idx == -1: continue

            #box_size = (rb[2]-rb[0])*(rb[3]-rb[1])
            # if box_size < 0.07: continue

            cx, cy = (rb[2]-rb[0])/2+rb[0], (rb[3]-rb[1])/2+rb[1]
            if not 0.1 < cx < 0.9 or not 0.1 < cy < 0.9:
                print('cx or cy out of bound', (cx, cy))
                continue

            rb_list.append(rb)
            vb_list.append(vector_bs)
            idx_list.append(search_idx)
            cate_list.append(cate)

    if len(rb_list) == 0:
        return "-1"
    res = Elk.multi_search_vec(Elk, idx_list, vb_list, vector.k, vector_type)
    es_res['es_res'] = es_parsing(res, rb_list, cate_list, multi=True)
    es_res['service_time'] = time.time() - search_time
    print('Interface time for searching vec', time.time()-search_time)

    return es_res


@app.post("/make_vector")
async def make_vector_db():
    debug = True
    save_time = time.time()
    sql = "SELECT * FROM `crawling_list` WHERE STATUS=1 AND cat_key NOT LIKE '%10'"
    product_list = imgtovec.connect_db(imgtovec, sql)
    print('Total count of product list', len(product_list))

    batch_size, split_data = 100, 1000
    img_path_list, cate_list, data_dict = [], [], {}

    for idx, product in enumerate(product_list):
        line = list(product)
        img_path = imgtovec.base_img_path + line[7] + os.sep + line[8]

        if not os.path.isfile(img_path): continue
        if cv2.imread(img_path) is None: continue
        try: Image.open(img_path).convert('RGB')
        except: continue

        img_path_list.append(img_path)
        cate_list.append(line[2])
        data_dict[img_path] = [line[0], line[1], line[2], line[3], line[5], line[6], img_path, line[9], line[10], line[15]]

        total_vec, total_list = {}, []
        if len(img_path_list) % split_data == 0:
            st_time = time.time()
            print(data_dict[img_path_list[-1]][0],'...ing')
            n = 0
            for size in range(math.ceil(len(img_path_list) / batch_size)):
                bulk_path = img_path_list[batch_size * n:batch_size * (n + 1)]
                category = cate_list[batch_size * n:batch_size * (n + 1)]
                n += 1

                vec = Yolov3.vector_extractor_by_model(Yolov3, bulk_path, category, batch_size)

                for idx in vec.keys():
                    if idx in total_vec.keys():
                        total_vec[idx] = total_vec[idx] + vec[idx]
                    else:
                        total_vec[idx] = vec[idx]

            if not debug: Elk.bulk_api_res_yolo(Elk, total_vec, data_dict)
            img_path_list.clear(), cate_list.clear(), data_dict.clear()

            print('# Interface time for saving 1000 data', time.time() - st_time)

    print('Interface time for saving all data', time.time() - save_time)

    return "send to elk"


@app.post("/make_vector_test")
async def make_vector_db_testset():
    debug=False
    save_time = time.time()

    product_list = glob.glob("testset/*.*")
    print(product_list)

    total_vec = {}
    for idx, img_path in enumerate(product_list):
        if not os.path.isfile(img_path): continue
        if cv2.imread(img_path) is None: continue
        try: Image.open(img_path).convert('RGB')
        except: continue

        with open(img_path, "rb") as f:
            img_b64 = base64.b64encode(f.read())

        vec = Yolov3.vector_extraction_service_testset(Yolov3, img_b64, img_path)

        for idx in vec.keys():
            if idx in total_vec.keys():
                total_vec[idx] = total_vec[idx] + vec[idx]
            else:
                total_vec[idx] = vec[idx]

    print(len(total_vec))
    if not debug: Elk.bulk_api_testset(Elk, total_vec)
    print('Interface time for saving all data', time.time() - save_time)

    return "send to elk"

img2vec = Image2Vector(cuda=True)

@app.post("/search_exist")
async def search_exist(d_img: DoubleIMG):
    total_time = time.time()
    save_list = []
    b64_img, c_key, img_url = d_img.img_b64, d_img.c_key, d_img.img_url
    # base64 to PIL image....
    img = Image.open(BytesIO(base64.b64decode(b64_img))).convert('RGB')

    st = time.time()
    vec = img2vec.get_vector(img)
    save_list += [str(time.time() - st)] #1. get_vector_time

    st = time.time()

    res = Elk.search_vec(Elk, "search_img", encode_array(vec))
    save_list += [str(time.time() - st)] #2. search_vec_time


    # initialize
    status = False
    if json.loads(res)['hits']['hits']==[]:
        Elk.save_vector_ps(Elk, vec, c_key, img_url)
        return status

    # image check process
    try:
        max_score = json.loads(res)['hits']['hits'][0]['_score']
        c_key = json.loads(res)['hits']['hits'][0]['_source']['img_url']
        print(c_key)
        if max_score> 0.98:
            status = True
            save_list += [str(time.time() - total_time)] #3. total_time
            return status
        else:
            status = False
            return status
    finally:
        if status is False:
            Elk.save_vector_ps(Elk, vec, c_key, img_url)
            save_list += [str(time.time() - total_time)]

        save_list += [img_url, str(status), str(max_score), target_url]
        save_data(save_list)


@app.post("/search_exist_debug")
async def search_exist(d_img: DoubleIMG):
    save_list = []
    b64_img, c_key, img_url = d_img.img_b64, d_img.c_key, d_img.img_url

    # base64 to PIL image....
    img = Image.open(BytesIO(base64.b64decode(b64_img))).convert('RGB')
    img2vec = Image2Vector()
    st = time.time()
    vec = img2vec.get_vector(img)
    save_list += [str(time.time() - st)]

    res = Elk.search_vec(Elk, "search_img", encode_array(vec))
    return res


@app.post("/yolo/searchVec_full")
async def search_vector_full(vector: Vector):
    Elk.search_index = 'test_v2_wc13'
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













