import json
import os

with open("DBinfo.json") as f:
    database = json.load(f)


# multiline 주석 : shift + alt + A

class DataConfig():
    CATE: str = "WC13"
    NUM: int = 65000

class YoloConfig():

    BATCH_SIZE: int = 100
    CONF_THRES: float = 0.2 
    NMS_THRES: float = 0.6

    # NOT CHANGE
    CFG: str = '/fastapi/yolov3/cfg/fashion/fashion_c23.cfg'
    WEIGHTS: str = '/mnt/piclick/piclick.ai/weights/best.pt'
    LABEL_PATH: str = '/fastapi/yolov3/data/fashion/fashion_c23.names'


class ExtractorCofing():
    DIM: int = 512

class ResultFile():
    BASE_DIR: str = "resultFile/v1/"

    # NOT CHANGE
    FVECS_BIN: str = "fvecs.bin"
    FNAMES: str = "fnames.txt"
    INDEX_TYPE: str = "hnsw"


class History():
    v1: str = "test trial"

class Setting():
    # Base Img Path
    BASE_IMG_PATH: str = "/mnt/piclick/piclick.tmp/AIMG/"

    # DB Auth info
    DATABASE_HOST: str = database['db']['host']
    DATABASE_USER: str = database['db']['user']
    DATABASE_PWD: str = database['db']['password']
    DATABASE_DB: str = database['db']['db']

    # Elastic info
    ELA_HOST: str = database['ela']['host']
    ELA_PORT: str = database['ela']['port']
    ELA_ADDR: str = ELA_HOST+":"+ELA_PORT