import torch

import time
from torchvision import transforms
import pickle
#import resnet.utils_ckb as utils_ckb
import utils_ckb
import torch.nn as nn
from PIL import Image

dparallel = False  # False 1 GPU,
batch_size = 2  # 512
num_classes = 23

model_path = "/fastapi/resnet/resnet_irs_v5"   # trained model
path_test = "/home/img_test"  # Folder : Input images

if torch.cuda.is_available():
    model = torch.load(model_path)
else:
    model = torch.load(model_path, map_location=torch.device('cpu'))


if dparallel :   # Send the model to multiple GPU
    if torch.cuda.device_count() :
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)

    if torch.cuda.is_available():
        model_ft = model.cuda()
else :
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Let's use 1 GPU - ", device)
    # Send the model to GPU
    model_ft = model.to(device)

model.eval()

input_size = 224
preprocess = transforms.Compose([
    transforms.Resize(input_size),
    transforms.CenterCrop(input_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

im_num = 0  # total number of images
match_num = 0

since = time.time()

image_datasets=utils_ckb.ImageFolder_ckb(path_test, preprocess)
dataloaders_test =torch.utils.data.DataLoader(image_datasets, batch_size = batch_size, shuffle=False, num_workers=4)

batch_no=2
# Iterate over data.

for inputs, labels, path in dataloaders_test:
    batch_no +=1
    if torch.cuda.is_available():
        if dparallel:
            inputs = inputs.cuda()
        else:
            inputs = inputs.to(device)
        # zero the parameter gradients
    with torch.no_grad():
        outputs = model(inputs)
        print(outputs.shape)

        if not dparallel :
            print(" vec output size : ", model.vec_out.shape)
            print(" filename path :  ", path)


