import time

from app.extVector import Vector
from app.extFaiss import FaissS

from app.dataBase import connectDB
from app.baseUtils import baseUtils

VEC = Vector()
faisss = FaissS()
util = baseUtils()

def makeDBindex(cate, num, writeM):

    flist = connectDB.getData(cate, num)
    yoloResult = VEC.getYoloBox(flist)

    for key in yoloResult['info'].keys():
        count = len(yoloResult['info'][key])
        st_time = time.time()
        

        # Save INFO
        util.saveInfos(yoloResult['info'][key], key, writeM)

        # Save Vector
        tensorStack = VEC.changePILtoTensorStack(yoloResult['vecs'][key])
        extVecs = VEC.extractVec(tensorStack)

        util.saveVecs(extVecs, key, writeM)

        # create Faiss Index
        faisss.writeIndexFile(key)

        endTime = time.time()-st_time

        print(f'[{key}] count : {count} & time : {endTime}')
        del st_time, endTime, count, tensorStack, extVecs

