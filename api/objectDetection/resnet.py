import torch
import pandas as pd
import json
from sklearn.random_projection import SparseRandomProjection

from yolov3.models import *
from yolov3.pirs_utils_v2 import *
from torch.utils.data import DataLoader

from PIL import Image
import torch
from torchvision import transforms


class Resnet:
    print("Loading resnet model ... ")
    resnet_path = "resnet/resnet_irs_v5"
    resnet_model = torch.load(resnet_path)
    resnet_model.eval()

    def get_vector(self, rawbox, img_path, svc=False):
        if svc:
            input_image = Image.fromarray(img_path,mode="RGB")
        else:
            input_image = Image.open(img_path)

        width, height = input_image.size
        crop_image = input_image.crop(rawbox)
        crop_image = crop_image.convert('RGB')

        input_size = 224
        preprocess = transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        input_tensor = preprocess(crop_image)
        input_batch = input_tensor.unsqueeze(0)  # create a mini-batch as expected by the model

        if torch.cuda.is_available():
            input_batch = input_batch.to('cuda')
            self.model.to('cuda')

        with torch.no_grad():
            self.model(input_batch)

        fv = self.model.vec_out[0].cpu().numpy()
        return fv