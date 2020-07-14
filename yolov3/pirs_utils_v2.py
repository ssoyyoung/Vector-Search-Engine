import numpy as np
import cv2
from io import BytesIO
import base64

import torch
import skimage.measure

import torch.nn.functional as F
from torch.utils.data import Dataset
from .utils.utils import *

img_formats = ['.bmp', '.jpg', '.jpeg', '.png', '.tif', '.dng']

def defaultbox(im0shape, r=0.1):
    h,w = im0shape
    tl = (int(w*r), int(h*r))
    br = (int(w*(1-r)), int(h*(1-r)))
    rb = (tl,br)
    return rb

def letterbox(img, new_shape=416, color=(128,128,128), mode='square'):
    # Resize a rectangular image to a 32 pixel multiple rectangle
    # https://github.com/ultralytics/yolov3/issues/232
    shape = img.shape[:2]  # current shape [height, width]

    if isinstance(new_shape, int):
        ratio = float(new_shape) / max(shape)
    else:
        ratio = max(new_shape) / max(shape)  # ratio  = new / old

    ratiow, ratioh = ratio, ratio
    new_unpad = (int(round(shape[1] * ratio)), int(round(shape[0] * ratio)))

    # Compute padding https://github.com/ultralytics/yolov3/issues/232
    if mode is 'auto':  # minimum rectangle
        dw = np.mod(new_shape - new_unpad[0], 32) / 2  # width padding
        dh = np.mod(new_shape - new_unpad[1], 32) / 2  # height padding
    elif mode is 'square':  # square
        dw = (new_shape - new_unpad[0]) / 2  # width padding
        dh = (new_shape - new_unpad[1]) / 2  # height padding
    elif mode is 'rect':  # square
        dw = (new_shape[1] - new_unpad[0]) / 2  # width padding
        dh = (new_shape[0] - new_unpad[1]) / 2  # height padding
    elif mode is 'scaleFill':
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape, new_shape)
        ratiow, ratioh = new_shape / shape[1], new_shape / shape[0]

    if shape[::-1] != new_unpad:  # resize
        if ratio < 1:
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_AREA)  # INTER_AREA is better, INTER_LINEAR is faster
        else:
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LANCZOS4)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratiow, ratioh, dw, dh

# DataLoader Util
from PIL import Image
def custom_letterbox(im, desired_size = 416):
    """
    Input : PIL image
    '
    Output : Padded PIL Image
    """

    old_size = im.size
    ratio = float(desired_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])
    im = im.resize(new_size, Image.ANTIALIAS)
    new_im = Image.new("RGB", (desired_size, desired_size), color="gray")
    new_im.paste(im, ((desired_size-new_size[0])//2,
                        (desired_size-new_size[1])//2))
    return new_im


def l2n(x,eps=1e-6):
    return x / (torch.norm(x, p=2, dim=0, keepdim=True) + eps).expand_as(x)

def mac(x):
    return F.max_pool2d(x, (x.size(-2), x.size(-1)))

def spoc(x):
    return F.avg_pool2d(x, (x.size(-2), x.size(-1)))

def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(0), x.size(1))).pow(1./p)

def rmac(x, L=3, eps=1e-6):
    ovr = 0.4
    steps = torch.Tensor([2,3,4,5,6,7]) # possible regions for the long dimension
    W = x.size(0)
    H = x.size(1)

    w = min(W, H)
    w2 = math.floor(w / 2.0 - 1)

    b = (max(H, W) - w) / (steps - 1)
    (tmp, idx) = torch.min(torch.abs(((w ** 2 - w * b) / w ** 2) - ovr), 0)  # steps(idx) regions for long dimension

    # region overplus per dimension
    Wd = 0;
    Hd = 0;
    if H < W:
        Wd = idx.item() + 1
    elif H > W:
        Hd = idx.item() + 1

    v = F.max_pool2d(x, (x.size(-2), x.size(-1)))
    v = v / (torch.norm(v, p=2, dim=1, keepdim=True) + eps).expand_as(v)

    for l in range(1, L + 1):
        wl = math.floor(2 * w / (l + 1))
        wl2 = math.floor(wl / 2 - 1)

        if l + Wd == 1:
            b = 0
        else:
            b = (W - wl) / (l + Wd - 1)
        cenW = torch.floor(wl2 + torch.Tensor(range(l - 1 + Wd + 1)) * b) - wl2  # center coordinates
        if l + Hd == 1:
            b = 0
        else:
            b = (H - wl) / (l + Hd - 1)
        cenH = torch.floor(wl2 + torch.Tensor(range(l - 1 + Hd + 1)) * b) - wl2  # center coordinates

        for i_ in cenH.tolist():
            for j_ in cenW.tolist():
                if wl == 0:
                    continue
                R = x[:, :, (int(i_) + torch.Tensor(range(wl)).long()).tolist(), :]
                R = R[:, :, :, (int(j_) + torch.Tensor(range(wl)).long()).tolist()]
                vt = F.max_pool2d(R, (R.size(-2), R.size(-1)))
                vt = vt / (torch.norm(vt, p=2, dim=1, keepdim=True) + eps).expand_as(vt)
                v += vt
    return v

def max_pooling_tensor(t, norm=True):
    t = t.permute([2,0,1])
    t_mac = mac(t).squeeze()
    if norm:
        t_mac_l2norm = l2n(t_mac)
        return t_mac_l2norm
    else:
        return t_mac

def average_pooling_tensor(t, norm=True):
    t = t.permute([2,0,1])
    t_spoc = spoc(t).squeeze()
    if norm:
        t_spoc_l2norm = l2n(t_spoc)
        return t_spoc_l2norm
    else:
        return t_spoc

def transfer(path, mode):
    img_size = 416
    img0 = cv2.imread(path)
    img = letterbox(img0, new_shape=img_size, mode=mode)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img, dtype=np.float32)
    img /= 255.0
    

    return img, img0

def transfer_cgd(img, mode="square"):
    if len(img.shape) == 2:
        img = np.stack((img,), axis=-1)

    img = img[:,:, [2,1,0]]
    img, ratiow, ratioh, dw, dh = letterbox(img, new_shape=224, mode="square")
    img = img.transpose(2, 0, 1)
    img = np.ascontiguousarray(img, dtype=np.float32)
    img /= 255.0
    
    img = torch.from_numpy(img)
    img = img.unsqueeze(0)  

    return img

def transfer_b64(decode_img, mode):
    img_size = 416
    img0 = decode_img
    img = letterbox(img0, new_shape=img_size, mode=mode)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img, dtype=np.float32)
    img /= 255.0

    return img, img0

def load_image(self, index):
    # loads 1 image from dataset
    img = self.imgs[index]
    if img is None:
        img_path = self.img_files[index]
        img = cv2.imread(img_path)  # BGR
        assert img is not None, 'Image Not Found ' + img_path
        r = self.img_size / max(img.shape)  # resize image to img_size
        if self.augment and (r != 1):  # always resize down, only resize up if training with augmentation
            h, w = img.shape[:2]
            return cv2.resize(img, (int(w * r), int(h * r)), interpolation=cv2.INTER_LINEAR)  # _LINEAR fastest
    return img

class LayerResult:
    def __init__(self, payers, layer_index):
        self.hook = payers[layer_index].register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.features_np = output.cpu().data.numpy()
        self.features = output

    def unregister_forward_hook(self):
        self.hook.remove()

class LoadImages(Dataset):
    def __init__(self, path, img_size=416, batch_size=16):
        self.img_files = [x for x in path if "." + x.split('.')[-1].lower() in img_formats]

        n = len(self.img_files)
        assert n > 0, 'No images found in %s' % (path)
        self.imgs = [None] * n
        bi = np.floor(np.arange(n) / batch_size).astype(np.int)  # batch index

        self.n = n
        #print("Total images {}".format(self.n))
        self.augment = False
        self.batch = bi  # batch index of image
        self.img_size = img_size

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        img_path = self.img_files[index]

        # Load image
        img0 = load_image(self, index)

        # Letterbox
        h0, w0 = img0.shape[:2]
        img, ratiow, ratioh, _, _ = letterbox(img0, new_shape=416, color=(128,128,128), mode='square')

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        return torch.from_numpy(img), img_path, (h0, w0)

    @staticmethod
    def collate_fn(batch):
        img, path, shapes = list(zip(*batch))  # transposed
        return torch.stack(img, 0), path, shapes


