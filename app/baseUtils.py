import os
import json
import struct
import numpy as np

from config import ResultConfig

class baseUtils():
    def __init__(self):
        # default settings 
        self.baseDir = ResultConfig.BASE_DIR
        self.fvecsBin = ResultConfig.FVECS_BIN
        self.fnamesTxt = ResultConfig.FNAMES


    def cleanUP(self, dirPath):
        '''
        Coment : delete all file in dirPath
        `
        @parameter : directory Path [Type : str]
        `
        @return : None
        '''
        if not os.path.exists(dirPath): 
            os.makedirs(dirPath)
        else:
            for file in os.scandir(dirPath):
                os.remove(file.path)

    def mkdirSavePath(self, key):
        '''
        Coment : create directory for saving data (if no directory, create new directory)
        `
        @parameter : category name [Type : str]
        `
        @return : new Save Path [Type: str]
        '''
        savePath = self.baseDir+key+"/"
        if not os.path.exists(savePath): os.makedirs(savePath)  

        return savePath   


    def saveVecs(self, vecs, key, writeM):
        '''
        Coment : create vec.bin file
        `
        @parameter : (1) stack of Tensor [Type : pytorch 4-dim tensor] > torch.Size([n, 512, 1, 1])
                     (2) key : folder_name, 
                     (3) write_method : [w: write new, a: write continue]
        `
        @return : None
        '''
        savePath = self.mkdirSavePath(key) + self.fvecsBin
              
        with open(savePath, writeM+'b') as f:
            fmt = f'{np.prod(vecs.shape)}f'
            f.write(struct.pack(fmt, *(vecs.flatten())))

    
    def saveInfos(self, info, key, writeM):
        '''
        Coment : create data_info.txt file
        `
        @parameter : (1) crop Img label [Type : list of dict] {'img_path','raw_box','conf','class'}, 
                     (2) key : folder_name, 
                     (3) write_method : [w: write new, a: write continue]
        `
        @return : None
        '''

        savePath = self.mkdirSavePath(key) + self.fnamesTxt

        with open(savePath, writeM) as f:
            for line in info:
                f.write(json.dumps(line)+'\n')

    
    def saveNumpy(self, nparrays, fileNames):
        if len(fileNames) == 1:
            np.save(fileNames, nparrays)
        else:
            for fileName, nparray in zip(fileNames, nparrays):
                np.save(fileName, nparray)