import os
import math
import glob
import time
import faiss
import random
import numpy as np
from sklearn.preprocessing import normalize

from config import ExtractorCofing, ResultConfig, DataConfig


# default settings 
baseDir = ResultConfig.BASE_DIR
fvecsBin = ResultConfig.FVECS_BIN
fnamesTxt = ResultConfig.FNAMES
indexType = ResultConfig.INDEX_TYPE


def get_index(index_type, dim):
    if index_type == 'hnsw':
        m = 48
        index = faiss.IndexHNSWFlat(dim, m)
        index.hnsw.efConstruction = 128
        return index
    elif index_type == 'l2':
        return faiss.IndexFlatL2(dim)
        
    raise

def populate(index, fvecs, batch_size=1000):
    nloop = math.ceil(fvecs.shape[0] / batch_size)
    for n in range(nloop):
        s = time.time()
        index.add(normalize(fvecs[n * batch_size : min((n + 1) * batch_size, fvecs.shape[0])]))
        print(n * batch_size, time.time() - s)

    return index

def getCate():
    fileList = glob.glob(baseDir+"*.txt")
    cateList = [f.split("/")[-1].split("_")[0]+"_" for f in fileList]
    return cateList

def createQueryDB():
    cateList = getCate()
    
    for cate in cateList:
        dim = ExtractorCofing.DIM
        fvec_file = baseDir + cate + fvecsBin
        index_file = f'{fvec_file}.{indexType}.index'

        fvecs = np.memmap(fvec_file, dtype='float32', mode='r').view('float32').reshape(-1, dim)

        if os.path.exists(index_file):
            index = faiss.read_index(index_file)
            if indexType == 'hnsw':
                index.hnsw.efSearch = 256
        else:
            index = get_index(indexType, dim)
            index = populate(index, fvecs)
            faiss.write_index(index, index_file)        

    return "create"


if __name__ == "__main__":
    createQueryDB()