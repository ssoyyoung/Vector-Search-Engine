import os
import math
import time
import faiss
import numpy as np
from sklearn.preprocessing import normalize

from app.baseUtils import baseUtils
from config import ResultConfig, ExtractorCofing, FaissConfig


class FaissS():
    def __init__(self):
        # default settings 
        self.baseDir = ResultConfig.BASE_DIR
        self.fvecsBin = ResultConfig.FVECS_BIN
        self.fnamesTxt = ResultConfig.FNAMES
        self.indexType = ResultConfig.INDEX_TYPE

        self.dim = ExtractorCofing.DIM
        self.util = baseUtils()

        # Search Faiss
        self.k = FaissConfig.K

    
    def get_index(self, dim):
        m = 48
        index = faiss.IndexHNSWFlat(dim, m)
        index.hnsw.efConstruction = 128

        return index
    
    def populate(self, index, fvecs, batch_size=1000):
        nloop = math.ceil(fvecs.shape[0] / batch_size)

        for n in range(nloop):
            s = time.time()
            index.add(normalize(fvecs[n * batch_size : min((n + 1) * batch_size, fvecs.shape[0])]))

        return index

    def defineIndexFile(self, key):
        indexFile = f'{self.baseDir}{key}/{self.indexType}.index'

        return indexFile

    def writeIndexFile(self, key):
        savePath = self.util.mkdirSavePath(key)
        fvecsFile = savePath+self.fvecsBin
        indexFile = self.defineIndexFile(key)

        fvecs = np.memmap(fvecsFile, dtype='float32', mode='r').view('float32').reshape(-1, self.dim)
        index = self.get_index(self.dim)
        index = self.populate(index, fvecs)
        faiss.write_index(index, indexFile)

        return index
    
    def loadIndexFile(self, indexFile):
        index = faiss.read_index(indexFile)
        index.hnsw.efSearch = 256

        return index


    def searchFaissIndex(self, key, vec):
        indexFile = self.defineIndexFile(key)

        if os.path.exists(indexFile): 
            index = self.loadIndexFile(indexFile)
        else:
            index = self.writeIndexFile(key)

        _, idxs = index.search(normalize(vec), self.k)

        return idxs

    