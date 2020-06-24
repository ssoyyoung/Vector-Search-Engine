import torch
import numpy as np
from PIL import Image

from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from vectorAPI.yoloModel import utils

# tranforms
transform = transforms.Compose([
        transforms.ToTensor()
    ])  

class YoloDataset(Dataset):
    def __init__(self, path, label, img_size, batch_size, transform):
        self.img_files = path
        self.label = label
        self.n = len(self.img_files)
        assert self.n > 0, 'No images found in %s' % (path)
        self.transform = transform
        self.imgs = [None] * self.n
        self.batch = np.floor(np.arange(self.n) / batch_size).astype(np.int)  # batch index of image
        self.img_size = img_size

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        '''
        Output : tensor, originalImg(PIL), imgPath, originalImg w & h
        '''

        # PIL image.size = width, height
        imgPath = self.img_files[idx]
        print(imgPath)
        img0 = Image.open(imgPath).convert('RGB')
        w0, h0 = img0.size
        img = utils.letterbox(img0, 416)
        
        img = self.transform(img)
        return img, img0, imgPath, self.label, (h0, w0)

    @staticmethod
    def collate_fn(batch):
        img, img0, path, label, shapes = list(zip(*batch))  # transposed
        return torch.stack(img, 0), img0, path, label, shapes

class YoloImg():
    def __init__(self, fname):
        self.img0 = Image.open(fname).convert('RGB')
        self.img = transform(utils.letterbox(self.img0, 416))
        self.path = fname
        self.shape = (self.img0.size[1], self.img0.size[0])
