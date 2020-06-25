import os
import cv2
import pymysql
from PIL import Image

from config import Setting

def connect():
    '''
    Commnet : Connect MySQL DataBase
    `
    @parameter : None
    `
    @return : Connection
    '''
    conn = pymysql.connect( 
                        host=Setting.DATABASE_HOST,
                        user=Setting.DATABASE_USER,
                        password=Setting.DATABASE_PWD,
                        db=Setting.DATABASE_DB
                        )

    return conn


def getData(cate, num):
    '''
    Commnet : Connect MySQL DB and get IMG list
    `
    @parameter : category(select db category), num(select data count(=limit))
    `
    @return : Image Path List [Type : List]
    '''

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
    imgList = checkImgStatus(imgList) #yoloModel.utils.checkImgStatus

    curs.close(), conn.close()

    return imgList



def checkImgStatus(imgList):
    '''
    Comnet : Image file Validation check
    `
    @parameter : Image Path [Type : List]
    `
    @return : Valid Image Path [Type : List]
    '''
    
    TimgList = []

    for img_path in imgList:
        if not os.path.isfile(img_path): 
            continue
        if cv2.imread(img_path) is None: 
            continue
        try: 
            Image.open(img_path).convert('RGB')
        except: 
            continue
        TimgList.append(img_path)

    return TimgList