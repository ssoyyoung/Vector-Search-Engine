import os
import cv2
import pymysql
from PIL import Image

from config import Setting

def connect():
    conn = pymysql.connect( 
                        host=Setting.DATABASE_HOST,
                        user=Setting.DATABASE_USER,
                        password=Setting.DATABASE_PWD,
                        db=Setting.DATABASE_DB
                        )

    return conn


def getData(cate, num):

    base_img_path = Setting.BASE_IMG_PATH

    conn = connect()

    if cate == "all":
        sql_query = "SELECT id, save_path, save_name FROM `crawling_list` WHERE STATUS=1 AND LIMIT "+str(num)
    else:
        sql_query = "SELECT id, save_path, save_name FROM `crawling_list` WHERE STATUS=1 AND cat_key='"+cate+"' LIMIT "+str(num)

    curs = conn.cursor()
    curs.execute(sql_query)
    data = curs.fetchall()

    # change img_path
    imgList = [base_img_path+path+"/"+name for _, path, name in data]
    idList = [dbId for dbId, _, _ in data]
    imgList = checkImgStatus(imgList,idList) #yoloModel.utils.checkImgStatus

    curs.close(), conn.close()

    return imgList



def checkImgStatus(imgList, idList):
    TimgList, TidList = [], []
    for img_path, dbId in zip(imgList,idList):
        if not os.path.isfile(img_path): 
            continue
        if cv2.imread(img_path) is None: 
            continue
        try: 
            Image.open(img_path).convert('RGB')
        except: 
            continue
        TimgList.append(img_path)
        TidList.append(dbId)

    return TimgList, TidList