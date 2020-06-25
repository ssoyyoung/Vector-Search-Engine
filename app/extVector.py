import glob
import struct
import json
import numpy as np
from PIL import Image

from config import YoloConfig, ResultConfig, DataConfig
from app.dataBase import connectDB
from app.yoloModel.dataloader import YoloDataset, YoloImg
from app.yoloModel import utils
from app.baseUtils import baseUtils

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from collections import defaultdict

# Model
from app.yoloModel.models import YoloV3
from app.baseModel.models import Resnet18

# yoloM = YoloV3()
# resnet18 = Resnet18()

class Vector():
    def __init__(self):
         # default settings 
        self.conf_thres = YoloConfig.CONF_THRES
        self.nms_thres = YoloConfig.NMS_THRES
        self.batch_size = YoloConfig.BATCH_SIZE
        self.image_size = YoloConfig.IMG_SIZE

        self.cate = DataConfig.CATE
        self.num = DataConfig.NUM

        # tranforms
        self.transform = transforms.Compose([
                transforms.ToTensor()
            ])    

        # Model
        self.yoloM = YoloV3()
        self.resnet18 = Resnet18()

        # Util 
        self.utils = baseUtils()


    def getYoloBox(self, fnameList):
        '''
        @parameter : image Path [Type : List]
        `
        @return : crop img tensor vector & info [Type : dict]
               >> return['info'][label] = [{'img_path', 'raw_box', 'conf', 'class'}]
                        ['vecs'][label] = [PIL.Image : RGB mode]   
        '''

        fnames = fnameList

        yoloDataset = YoloDataset(
                            path=fnames,     
                            label=[], 
                            img_size=self.image_size, 
                            batch_size=self.batch_size, 
                            transform=self.transform)

        yoloDataLoader = DataLoader(
                            yoloDataset, 
                            batch_size=self.batch_size,
                            shuffle=False, 
                            num_workers=1,
                            collate_fn=yoloDataset.collate_fn)
        
        # create defaultdict for return
        res = defaultdict(lambda: defaultdict(list))

        # iterate for box data
        for batch, (imgs, img0s, paths, labels, shapes) in enumerate(yoloDataLoader):
            print(f'######## {batch} ########')

            torch.cuda.empty_cache()

            with torch.no_grad():   
                imgs = imgs.to(self.yoloM.device)   
                _, _, height, width = imgs.shape        

                pred = self.yoloM.model(imgs)[0]
                pred = utils.non_max_suppression(pred, conf_thres=self.conf_thres, nms_thres=self.nms_thres)

            for i, det in enumerate(pred):
                img0shape = shapes[i]
                if det is not None and len(det):
                    # make original size
                    det[:, :4] = utils.scale_coords(imgs.size()[2:], det[:, :4], img0shape).round() 

                    for *xyxy, conf, lab in det:
                        saveLabel = self.yoloM.names[int(lab)]
                        xyxy = [int(x) for x in xyxy]

                        res['info'][saveLabel].append({
                                                    "img_path" : paths[i], 
                                                    "raw_box" : xyxy, 
                                                    "conf" : round(float(conf),3), 
                                                    "class" : saveLabel
                                                    })
                        
                        cropBox = utils.letterbox(img0s[i].crop(xyxy), desired_size=224)
                        res['vecs'][saveLabel].append(cropBox)

        return res
    
    def changePILtoTensorStack(self, PILlist):
        '''
        @parameter : PIL image list [Type : List of PIL image]
        `
        @return : stack of Tensor [Type : 4-dim tensor]
        '''
        tensorStack = torch.stack([self.transform(PIL) for PIL in PILlist], 0)
        
        return tensorStack


    def extractVec(self, vecs):
        '''
        @parameter : stack of Tensor [Type : pytorch 4-dim tensor] > torch.Size([n, 3, 224, 224])
        `
        @return : stack of Tensor [Type : pytorch 4-dim tensor] > torch.Size([n, 512, 1, 1])
        '''

        vecs = self.resnet18.FE(vecs)

        return vecs            

