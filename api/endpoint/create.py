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

router = APIRouter()


@router.put("/db/{cate}")
async def make_vector_db(cate: str):
    debug = True
    save_time = time.time()
    
    if cate == "all":
        sql = "SELECT * FROM `crawling_list` WHERE STATUS=1 AND cat_key NOT LIKE '%10'"
    else:
        sql = "SELECT * FROM `crawling_list` WHERE STATUS=1 AND cat_key='"+cate+"'"
    print(sql)
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


@router.post("/testset")
async def make_vector_db_testset():
    debug=True
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