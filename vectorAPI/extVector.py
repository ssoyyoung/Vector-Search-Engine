import os
import glob
import struct
import json
import numpy as np
from PIL import Image

from config import YoloConfig, ResultFile, DataConfig
from vectorAPI.dataBase import connectDB
from vectorAPI.yoloModel.dataloader import YoloDataset, YoloImg
from vectorAPI.yoloModel import utils

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from collections import defaultdict

# Model
from vectorAPI.yoloModel.models import YoloV3
from vectorAPI.baseModel.models import Resnet18

yoloM = YoloV3()
resnet18 = Resnet18()

# default settings 
baseDir = ResultFile.BASE_DIR
fvecsBin = ResultFile.FVECS_BIN
fnamesTxt = ResultFile.FNAMES

conf_thres = YoloConfig.CONF_THRES
nms_thres = YoloConfig.NMS_THRES

cate = DataConfig.CATE
num = DataConfig.NUM

# tranforms
transform = transforms.Compose([
        transforms.ToTensor()
    ])    

def cleanUP(dirName):
    if not os.path.exists(dirName): 
        os.makedirs(dirName)
    else:
        for file in os.scandir(dirName):
            os.remove(file.path)

def extractVec(fname):
    Input = YoloImg(fname)

    with torch.no_grad():   
        img = Input.img.to(yoloM.device)   
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        pred = yoloM.model(img)[0]
        pred = utils.non_max_suppression(pred, conf_thres=conf_thres, nms_thres=nms_thres)
   
        res = defaultdict(lambda: defaultdict(list))
        for i, det in enumerate(pred):
            img0shape = Input.shape
            if det is not None and len(det):
                # make original size
                det[:, :4] = utils.scale_coords(img.size()[2:], det[:, :4], img0shape).round() 

                for *xyxy, conf, lab in det:
                    xyxy = [int(x) for x in xyxy]
                    res[yoloM.names[int(lab)]]['info'].append({
                                                                "img_path" : Input.path, 
                                                                "raw_box" : xyxy, 
                                                                "conf" : round(float(conf),3), 
                                                                "class" : yoloM.names[int(lab)]
                                                                })
                    
                    cropBox = utils.letterbox(Input.img0.crop(xyxy), desired_size=224)
                    res[yoloM.names[int(lab)]]['res'].append(cropBox)

                for key in res.keys():
                    res[key]['res'] = torch.stack([transform(x) for x in res[key]['res']],0)
                
    torch.cuda.empty_cache()
    return res


def extractVecAndSave():
    cleanUP(baseDir)

    batch_size = YoloConfig.BATCH_SIZE

    fnames, ids = connectDB.getData(cate = cate, num = num)


    print("Total dataset count...", fnames)

    yoloDataset = YoloDataset(
                            path=fnames,     
                            label=[], 
                            img_size=416, 
                            batch_size=batch_size, 
                            transform=transform)

    yoloDataLoader = DataLoader(
                            yoloDataset, 
                            batch_size=batch_size,
                            shuffle=False, 
                            num_workers=1,
                            collate_fn=yoloDataset.collate_fn)


    for batch, (imgs, img0s, paths, labels, shapes) in enumerate(yoloDataLoader):
        tmpID = ids[batch*100:(batch+1)*100]

        torch.cuda.empty_cache()

        with torch.no_grad():   
            imgs = imgs.to(yoloM.device)   
            _, _, height, width = imgs.shape        

            pred = yoloM.model(imgs)[0]
            pred = utils.non_max_suppression(pred, conf_thres=conf_thres, nms_thres=nms_thres)

        #info, res = [], []
        res = defaultdict(lambda: defaultdict(list))
        for i, det in enumerate(pred):
            img0shape = shapes[i]
            if det is not None and len(det):
                # make original size
                det[:, :4] = utils.scale_coords(imgs.size()[2:], det[:, :4], img0shape).round() 

                for *xyxy, conf, lab in det:
                    xyxy = [int(x) for x in xyxy]
                    #info.append([paths[i], xyxy, round(float(conf),3), yoloM.names[int(lab)]])
                    res[yoloM.names[int(lab)]]['info'].append({
                                                                "img_path" : paths[i], 
                                                                "raw_box" : xyxy, 
                                                                "conf" : round(float(conf),3), 
                                                                "class" : yoloM.names[int(lab)],
                                                                "id" : tmpID[i]
                                                                })
                    
                    cropBox = utils.letterbox(img0s[i].crop(xyxy), desired_size=224)
                    #res.append(cropBox)
                    res[yoloM.names[int(lab)]]['res'].append(cropBox)
            
        for key in res.keys():
            newBaseDir = baseDir+key+"_"
            result = torch.stack([transform(x) for x in res[key]['res']], 0)
            print(key, ">>", result.shape)

            with open(newBaseDir+fvecsBin, 'ab') as f:
                fvecs = resnet18.FE(result)

                fmt = f'{np.prod(fvecs.shape)}f'
                f.write(struct.pack(fmt, *(fvecs.flatten())))

            with open(newBaseDir+fnamesTxt, 'a') as f:
                for line in res[key]['info']:
                    f.write(json.dumps(line)+'\n')
        
        print("Done")

            


""" def extractVecAndSave(yoloM, resnet18):
    batch_size = YoloConfig.BATCH_SIZE

    fnames, ids = connectDB.getData(cate = "WC13", num = 10)
    print("Total dataset count...", fnames)

    yoloDataset = YoloDataset(
                            path=fnames,     
                            label=[], 
                            img_size=416, 
                            batch_size=batch_size, 
                            transform=transform)

    yoloDataLoader = DataLoader(
                            yoloDataset, 
                            batch_size=batch_size,
                            shuffle=False, 
                            num_workers=1,
                            collate_fn=yoloDataset.collate_fn)


    for batch, (imgs, img0s, paths, labels, shapes) in enumerate(yoloDataLoader):
        tmpID = ids[batch*100:(batch+1)*100]

        torch.cuda.empty_cache()

        with torch.no_grad():   
            imgs = imgs.to(yoloM.device)   
            _, _, height, width = imgs.shape        

            pred = yoloM.model(imgs)[0]
            pred = utils.non_max_suppression(pred, conf_thres=conf_thres, nms_thres=nms_thres)

        info, res = [], []
        for i, det in enumerate(pred):
            img0shape = shapes[i]
            if det is not None and len(det):
                # make original size
                det[:, :4] = utils.scale_coords(imgs.size()[2:], det[:, :4], img0shape).round() 

                for *xyxy, conf, lab in det:
                    xyxy = [int(x) for x in xyxy]
                    info.append([paths[i], xyxy, round(float(conf),3), yoloM.names[int(lab)]])
                    
                    cropBox = utils.letterbox(img0s[i].crop(xyxy), desired_size=224)
                    res.append(cropBox)
        
        
        result = torch.stack([transform(x) for x in res], 0)

        if not os.path.exists(baseDir): os.makedirs(baseDir)
        if os.path.exists(baseDir+fvecsBin): os.remove(baseDir+fvecsBin)

        with open(baseDir+fvecsBin, 'ab') as f:
            fvecs = resnet18.FE(result)

            fmt = f'{np.prod(fvecs.shape)}f'
            f.write(struct.pack(fmt, *(fvecs.flatten())))
        print("Writing fvecs bin!")

        if os.path.exists(baseDir+fnamesTxt): os.remove(baseDir+fnamesTxt)

        with open(baseDir+fnamesTxt, 'a') as f:
            for line in info:
                f.write(json.dumps(line)+'\n')
        
        del result
 """