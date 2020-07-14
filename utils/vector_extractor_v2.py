import pandas as pd
import json
from sklearn.random_projection import SparseRandomProjection

from yolov3.models import *
from yolov3.pirs_utils_v2 import *
from torch.utils.data import DataLoader

from PIL import Image
import torch
from torchvision import transforms

try:
    from apex import amp
    mixed_precision = True
except:
    mixed_precision = False

with open('utils/category_mapping.json', 'r') as f:
    mapping = json.load(f)

with open('utils/category_mapping_search.json', 'r') as f:
    mapping_ops = json.load(f)

from cirtorch.networks.cgd import init_network, extract_vectors, extract_vectors_svc

import torch
from torch.utils.data import DataLoader
from torchvision import transforms, utils

class Cgd:
    model_params = {}
    model_params['architecture'] = 'seresnet50'
    model_params['output_dim'] = 1536
    model_params['combination'] = 'GS'
    model_params['pretrained'] = True
    model_params['classes'] = 7982

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # state62 = torch.load('/mnt/piclick/piclick.ai/weights/model62_best.pth.tar')
    # model62 = init_network(model_params).to(device)
    # model62.load_state_dict(state62['state_dict'])

    # state48 = torch.load('/mnt/piclick/piclick.ai/weights/model48_best.pth.tar')
    # model48 = init_network(model_params).to(device)
    # model48.load_state_dict(state48['state_dict'])

    #state10 = torch.load('/mnt/piclick/piclick.ai/weights/model10_best.pth.tar')
    #model10 = init_network(model_params).to(device)
    #model10.load_state_dict(state10['state_dict'])

    # state44 = torch.load('/mnt/piclick/piclick.ai/weights/model44_best.pth.tar')
    # model44 = init_network(model_params).to(device)
    # model44.load_state_dict(state44['state_dict'])

    # state90 = torch.load('/mnt/piclick/piclick.ai/weights/model90_best.pth.tar')
    # model90 = init_network(model_params).to(device)
    # model90.load_state_dict(state90['state_dict'])

    state7 = torch.load('/mnt/piclick/piclick.ai/weights/model7_best.pth.tar')
    model7 = init_network(model_params).to(device)
    model7.load_state_dict(state7['state_dict'])

    def get_cgd_vec(self, images):

        transform = transforms.Compose([
            transforms.Resize(252),
            #transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.4324, 0.4994, 0.4856),(0.1927, 0.1811, 0.1817))
            ])

        vecs = extract_vectors(self.model, images, image_size=None, transform=transform, bbxs=None, ms=[1], msp=1, print_freq=10)
        #vec_flat = torch.flatten(vecs).cpu().numpy()
        return vecs

    def get_cgd_vec_decodeImg(self, pilImage):

        transform = transforms.Compose([
            transforms.Resize(252),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.4324, 0.4994, 0.4856),(0.1927, 0.1811, 0.1817))
            ])

        vecs = extract_vectors_svc(self.model, [pilImage], image_size=None, transform=transform, bbxs=None, ms=[1], msp=1, print_freq=10)
        vec_flat = torch.flatten(vecs).cpu().numpy()
        return vec_flat    
    
    def get_cgd_vec_crop(self, xyxy, imgPath, model_num=62):

        img = Image.open(imgPath).convert('RGB').crop(xyxy)
        img = np.asarray(img, dtype=np.uint8)
        
        img = transfer_cgd(img)
        
        if model_num == 62:
            model = self.model62
        elif model_num == 48:
            model = self.model48

        vecs = extract_vectors_svc(model, img, image_size=None, transform=None, bbxs=None, ms=[1], msp=1, print_freq=10)
        vec_flat = torch.flatten(vecs).cpu().numpy()
        return vec_flat   
    
    # MAKE DB
    def get_cgd_vec_crop_batch(self, cgd_xyxy, cgd_img, model_num=62):
        
        cgd_img = [Image.open(imgPath).convert('RGB').crop(xyxy)for imgPath, xyxy in zip(cgd_img, cgd_xyxy)]

        PILlist = [custom_letterbox(img, desired_size=224) for img in cgd_img]

        transform = transforms.Compose([
                transforms.ToTensor()
            ])  

        tensorStack = torch.stack([transform(PIL) for PIL in PILlist], 0)
        
        if model_num == 62:
            model = self.model62
        elif model_num == 48:
            model = self.model48
        elif model_num == 10:
            model = self.model10
        elif model_num == 44:
            model = self.model44
        elif model_num == 90:
            model = self.model90
        elif model_num == 7:
            model = self.model7

        vecs = extract_vectors_svc(model, tensorStack, image_size=None, transform=None, bbxs=None, ms=[1], msp=1, print_freq=10)
        #vec_flat = torch.flatten(vecs).cpu().numpy()
        return vecs  
    
    # SEARCH IMG
    def get_cgd_vec_crop_batch_SVC(self, cgd_xyxy, decode_img, model_num=62):
        crop_img = Image.fromarray(decode_img).convert('RGB').crop(cgd_xyxy)
        
        #cgd_img = [Image.fromarray(dimg).convert('RGB').crop(xyxy)for dimg, (xyxy) in zip(decode_img, cgd_xyxy)]

        #PILlist = [custom_letterbox(img, desired_size=224) for img in crop_img]
        le_img = custom_letterbox(crop_img, desired_size=224)

        transform = transforms.Compose([
                transforms.ToTensor()
            ])  

        tensorStack = torch.stack([transform(le_img)], 0)
        
        if model_num == 62:
            model = self.model62
        elif model_num == 10:
            model = self.model10
        elif model_num == 44:
            model = self.model44
        elif model_num == 90:
            model = self.model90
        elif model_num == 7:
            model = self.model7

        vecs = extract_vectors_svc(model, tensorStack, image_size=None, transform=None, bbxs=None, ms=[1], msp=1, print_freq=10)
        #vec_flat = torch.flatten(vecs).cpu().numpy()
        return vecs  



class Resnet:

    model_path_1024 = "/mnt/piclick/piclick.ai/weights/resnet_irs_v9"
    model_1024 = torch.load(model_path_1024)
    model_1024.eval()
    print("loading 1024 resnet")

    model_path_512 = "/mnt/piclick/piclick.ai/weights/resnet_irs_v5"
    model_512 = torch.load(model_path_512)
    model_512.eval()
    print("loading 512 resnet")

    def get_feature_vector(self, rawbox, img_path, svc=False, vec="1024"):
        if svc:
            input_image = Image.fromarray(img_path,mode="RGB")
        else:
            input_image = Image.open(img_path)

        width, height = input_image.size
        crop_image = input_image.crop(rawbox)
        crop_image = crop_image.convert('RGB')

        if vec == "1024":
            model = self.model_1024
        elif vec == "512":
            model = self.model_512

        input_size = 224
        preprocess = transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4589, 0.4159, 0.4185] , std=[0.2328, 0.2218, 0.2162]),
        ])
        input_tensor = preprocess(crop_image)
        input_batch = input_tensor.unsqueeze(0)  # create a mini-batch as expected by the model

        if torch.cuda.is_available():
            input_batch = input_batch.to('cuda')
            model.to('cuda')

        with torch.no_grad():
            model(input_batch)

        fv = model.vec_out[0].cpu().numpy()
        print("done")
        return fv

class Yolov3:
    img_size = 416

    cfg = '/fastapi/yolov3/cfg/fashion/fashion_c23.cfg'
    weights = '/mnt/piclick/piclick.ai/weights/best.pt'
    names_path = '/fastapi/yolov3/data/fashion/fashion_c23.names'

    device = torch_utils.select_device('', apex=mixed_precision)
    model = Darknet(cfg, arc='default').to(device).eval()
    model.load_state_dict(torch.load(weights, map_location=device)['model'])
    names = load_classes(names_path)

    def vector_extractor_by_model(self, data, category, batch_size):
        names = load_classes(self.names_path)

        list_names = {}

        dataset = LoadImages(data, self.img_size, batch_size=batch_size)
        batch_size = min(batch_size, len(dataset))
        dataloader = DataLoader(dataset,
                                batch_size=batch_size,
                                num_workers=min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8]),
                                pin_memory=True,
                                collate_fn=dataset.collate_fn)

        # [batch, 3, 416, 416], [path1, path2], [[h,w,3],[h,w,3]]
        for batch_i, (imgs, paths, shapes) in enumerate(dataloader):
            torch.cuda.empty_cache()

            with torch.no_grad():
                imgs = imgs.to(self.device).float() / 255.0
                _, _, height, width = imgs.shape

                layerResult = LayerResult(self.model.module_list, 80)
                pred = self.model(imgs)[0]

                layerResult_tensor = layerResult.features.permute([0, 2, 3, 1])
                kernel_size = layerResult_tensor.shape[1]
                LayerResult.unregister_forward_hook(layerResult)

                # box info
                pred = non_max_suppression(pred, conf_thres=0.5, nms_thres=0.6)

            # Process detections
            for i, det in enumerate(pred):  # detections per image
                c = category[batch_i * batch_size + i]

                try:
                    state = list_names[c].copy()
                except:
                    list_names[c] = []
                    state = list_names[c].copy()

                im0shape = shapes[i]  # original shape

                if det is not None and len(det):
                    # 검출된 좌표가 0보다 작거나 416보다 클 경우
                    det = torch.clamp(det, min=0, max=416)

                    ratio = kernel_size / self.img_size
                    resized_det = det[:, :4] * ratio

                    feature_box = np.asarray(torch.round(resized_det).detach().cpu().numpy(), dtype=np.int32)
                    det[:, :4] = scale_coords(imgs.shape[2:], det[:, :4], im0shape).round()  # originial

                    for (*xyxy, conf, cls), fb in zip(det, feature_box):

                        # 직선형태로 박스가 잡혔다면,
                        if fb[0] == fb[2] or fb[1] == fb[3]: continue
                        if conf < 0.3: continue
                        # 카테고리가 맞지 않는 경우
                        if mapping[c[1:]] != names[int(cls)].lower():
                            # 혼합 카테고리가 들어온 경우
                            if c[2:] == "10":
                                # 하위 카테고리 안에 들어오지 않을 때
                                if c[1] != \
                                        [code for code, cate in mapping.items() if cate == names[int(cls)].lower()][0][
                                            0]: continue
                            # 다른 카테고리가 들어온 경우
                            else:
                                continue

                        feature_result = layerResult_tensor[i, fb[0]:fb[2] + 1, fb[1]:fb[3] + 1]

                        xyxy = [np.asarray(torch.round(x).cpu().numpy(), dtype=np.int32) for x in xyxy]

                        #RESNET = Resnet.get_feature_vector(Resnet, [int(x) for x in xyxy], paths[i])
                        CGD_62 = Cgd.get_cgd_vec_crop(Cgd, [int(x) for x in xyxy], paths[i], 62)
                        CGD_48 = Cgd.get_cgd_vec_crop(Cgd, [int(x) for x in xyxy], paths[i], 48)
                        
                        #MAC = max_pooling_tensor(feature_result)
                        #SPoC = average_pooling_tensor(feature_result)
                        #CAT = torch.cat((MAC, SPoC))

                        xyxy[0], xyxy[2] = xyxy[0] / im0shape[1], xyxy[2] / im0shape[1]
                        xyxy[1], xyxy[3] = xyxy[1] / im0shape[0], xyxy[3] / im0shape[0]

                        class_data = {
                            'raw_box': xyxy,
                        #    'yolo_vector_mac': MAC.detach().cpu().numpy(),
                        #    'yolo_vector_spoc': SPoC.detach().cpu().numpy(),
                        #    'yolo_vector_cat': CAT.detach().cpu().numpy(),
                        #    'resnet_vector': RESNET,
                            'cgd62_vector' : CGD_62,
                            'cgd48_vector' : CGD_48,
                            'img_path': paths[i],
                            'state': 'fbox'
                        }

                        list_names[c].append(class_data)


                # if no change
                if list_names[c] == state:
                    default_feature_result = layerResult_tensor[i, 1:12, 1:12]
                    data = defaultbox(im0shape, r=0.1)

                    xyxy = [round(d) for d in list(data[0]) + list(data[1])]

                    CGD_62 = Cgd.get_cgd_vec_crop(Cgd, [int(x) for x in xyxy], paths[i], 62)
                    CGD_48 = Cgd.get_cgd_vec_crop(Cgd, [int(x) for x in xyxy], paths[i], 48)

                    #RESNET = Resnet.get_feature_vector(Resnet, xyxy, paths[i])
                    #MAC = max_pooling_tensor(default_feature_result)
                    #SPoC = average_pooling_tensor(default_feature_result)
                    #SPoc_wo_norm = average_pooling_tensor(default_feature_result, False)
                    #CAT = torch.cat((MAC, SPoC))

                    xyxy[0], xyxy[2] = xyxy[0] / im0shape[1], xyxy[2] / im0shape[1]
                    xyxy[1], xyxy[3] = xyxy[1] / im0shape[0], xyxy[3] / im0shape[0]

                    class_data = {
                        'raw_box': xyxy,
                        #    'yolo_vector_mac': MAC.detach().cpu().numpy(),
                        #    'yolo_vector_spoc': SPoC.detach().cpu().numpy(),
                        #    'yolo_vector_cat': CAT.detach().cpu().numpy(),
                        #    'resnet_vector': RESNET,
                        'cgd62_vector' : CGD_62,
                        'cgd48_vector' : CGD_48,
                        'img_path': paths[i],
                        'state': 'dbox'
                        }
                    # class_data = {
                    #     'raw_box': xyxy,
                    #     'yolo_vector_mac': MAC.detach().cpu().numpy(),
                    #     'yolo_vector_spoc': SPoC.detach().cpu().numpy(),
                    #     'yolo_vector_spoc_w/o_norm': SPoc_wo_norm.detach().cpu().numpy(),
                    #     'yolo_vector_cat': CAT.detach().cpu().numpy(),
                    #     'resnet_vector': RESNET,
                    #     'img_path': paths[i],
                    #     'state': 'fbox'
                    # }

                    list_names[c].append(class_data)

            #print(" Inference time for batch image : {}".format(batch_end))

        return list_names

    def vector_extractor_by_model_service(self, base_img, type):
        img_res = {}
        list_names = []

        names = load_classes(self.names_path)

        for i in range(len(names)):
            list_names.append([])

        batch_time = time.time()
        torch.cuda.empty_cache()

        with torch.no_grad():
            img_bytes = base64.b64decode(base_img)
            file_bytes = np.asarray(bytearray(BytesIO(img_bytes).read()), dtype=np.uint8)
            decode_img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            img, img0 = transfer_b64(decode_img, mode='square')  # auto, square, rect, scaleFill / default : square
            img = torch.from_numpy(img).to(self.device)
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            layerResult = LayerResult(self.model.module_list, 80)

            pred = self.model(img)[0]

            layerResult_tensor = layerResult.features.permute([0, 2, 3, 1])
            kernel_size = layerResult_tensor.shape[1]

            LayerResult.unregister_forward_hook(layerResult)

            # box info
            pred = non_max_suppression(pred, conf_thres=0.5, nms_thres=0.5)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if det is not None and len(det):

                ratio = kernel_size / self.img_size
                resized_det = det[:, :4] * ratio

                im0shape = img0.shape
                feature_box = np.asarray(resized_det.detach().cpu().numpy(), dtype=np.int32)

                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0shape).round()  # originial

                for (*xyxy, conf, cls), fb in zip(det, feature_box):
                    if conf < 0.5: continue
                    feature_result = layerResult_tensor[i, fb[0]:fb[2] + 1, fb[1]:fb[3] + 1]

                    xyxy = [int(x) for x in xyxy]

                    if type == "resnet512":
                        RESNET = Resnet.get_feature_vector(Resnet, xyxy, decode_img, svc=True, vec="512")
                    elif type == "resnet1024":
                        RESNET = Resnet.get_feature_vector(Resnet, xyxy, decode_img, svc=True, vec="1024")
                    elif type == "cgd10":
                        CGD = Cgd.get_cgd_vec_crop_batch_SVC(Cgd, xyxy, decode_img, 10)
                    elif type == "cgd62":
                        CGD = Cgd.get_cgd_vec_crop_batch_SVC(Cgd, xyxy, decode_img, 62)
                    elif type == "cgd44":
                        CGD = Cgd.get_cgd_vec_crop_batch_SVC(Cgd, xyxy, decode_img, 44)    
                    elif type == "cgd90":
                        CGD = Cgd.get_cgd_vec_crop_batch_SVC(Cgd, xyxy, decode_img, 90)   
                    elif type == "cgd7":
                        CGD = Cgd.get_cgd_vec_crop_batch_SVC(Cgd, xyxy, decode_img, 7)   

                    MAC = max_pooling_tensor(feature_result)
                    SPoC = average_pooling_tensor(feature_result)
                    CAT = torch.cat((MAC, SPoC))

                    xyxy[0], xyxy[2] = xyxy[0] / im0shape[1], xyxy[2] / im0shape[1]
                    xyxy[1], xyxy[3] = xyxy[1] / im0shape[0], xyxy[3] / im0shape[0]

                    if type == 'mac':
                        vector = MAC.detach().cpu().numpy()
                    elif type == 'spoc':
                        vector = SPoC.detach().cpu().numpy()
                    elif type == 'cat':
                        vector = CAT.detach().cpu().numpy()
                    elif 'resnet' in type:
                        vector = RESNET
                    elif 'cgd' in type:
                        vector = CGD

                    class_data = {
                        'raw_box': xyxy,
                        'feature_vector': vector,
                        'category': names[int(cls.cpu().numpy())],
                        'img_size': im0shape
                    }

                    list_names[int(cls)].append(class_data)

        for i in range(len(names)):
            if list_names[int(i)] != []:
                img_res[names[int(i)]] = list_names[int(i)]

        batch_end = time.time() - batch_time
        print("Inference time for a image : {}".format(batch_end))

        return img_res


    def vector_extraction_batch(self, data, category, batch_size = 1, rp=False, pooling='max'):
        names = load_classes(self.names_path)

        list_names = {}

        dataset = LoadImages(data, self.img_size, batch_size=batch_size)
        batch_size = min(batch_size, len(dataset))
        dataloader = DataLoader(dataset,
                                batch_size=batch_size,
                                num_workers=min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8]),
                                pin_memory=True,
                                collate_fn=dataset.collate_fn)

        # [batch, 3, 416, 416], [path1, path2], [[h,w,3],[h,w,3]]
        for batch_i, (imgs, paths, shapes) in enumerate(tqdm(dataloader)):
            batch_time = time.time()
            torch.cuda.empty_cache()

            with torch.no_grad():
                imgs = imgs.to(self.device).float() / 255.0
                _, _, height, width = imgs.shape

                layerResult = LayerResult(self.model.module_list, 80)
                pred = self.model(imgs)[0]

                layerResult_tensor = layerResult.features.permute([0, 2, 3, 1])
                kernel_size = layerResult_tensor.shape[1]
                LayerResult.unregister_forward_hook(layerResult)

                # box info
                pred = non_max_suppression(pred, conf_thres=0.5, nms_thres=0.6)

            # Process detections
            for i, det in enumerate(pred):  # detections per image
                c = category[batch_i * batch_size + i]

                try:
                    state = list_names[c].copy()
                except:
                    list_names[c] = []
                    state = list_names[c].copy()

                im0shape = shapes[i]  # original shape

                if det is not None and len(det):
                    # 검출된 좌표가 0보다 작거나 416보다 클 경우
                    det = torch.clamp(det, min=0, max=416)

                    ratio = kernel_size / self.img_size
                    resized_det = det[:, :4] * ratio

                    feature_box = np.asarray(torch.round(resized_det).detach().cpu().numpy(), dtype=np.int32)
                    det[:, :4] = scale_coords(imgs.shape[2:], det[:, :4], im0shape).round()  # originial

                    for (*xyxy, conf, cls), fb in zip(det, feature_box):

                        # 직선형태로 박스가 잡혔다면,
                        if fb[0] == fb[2] or fb[1] == fb[3]: continue
                        if conf < 0.3: continue
                        # 카테고리가 맞지 않는 경우
                        if mapping[c[1:]] != names[int(cls)].lower():
                            # 혼합 카테고리가 들어온 경우
                            if c[2:] == "10":
                                # 하위 카테고리 안에 들어오지 않을 때
                                if c[1] != [code for code, cate in mapping.items() if cate == names[int(cls)].lower()][0][0]: continue
                            # 다른 카테고리가 들어온 경우
                            else: continue

                        feature_result = layerResult_tensor[i, fb[0]:fb[2] + 1, fb[1]:fb[3] + 1]

                        xyxy = [np.asarray(torch.round(x).cpu().numpy(), dtype=np.int32) for x in xyxy]

                        xyxy[0], xyxy[2] = xyxy[0] / im0shape[1], xyxy[2] / im0shape[1]
                        xyxy[1], xyxy[3] = xyxy[1] / im0shape[0], xyxy[3] / im0shape[0]

                        class_data = {
                            'raw_box': xyxy,
                            'feature_vector': max_pooling_tensor(feature_result) if pooling == 'max' else average_pooling_tensor(feature_result),
                            'img_path': paths[i],
                            'state' : 'fbox'
                        }

                        list_names[c].append(class_data)

                # if no change
                if list_names[c] == state:

                    default_feature_result = layerResult_tensor[i, 1:12,1:12]
                    data = defaultbox(im0shape, r =0.1)

                    xyxy = [round(d) for d in list(data[0]) + list(data[1])]
                    xyxy[0], xyxy[2] = xyxy[0] / im0shape[1], xyxy[2] / im0shape[1]
                    xyxy[1], xyxy[3] = xyxy[1] / im0shape[0], xyxy[3] / im0shape[0]

                    class_data = {
                        'raw_box': xyxy,
                        'feature_vector': max_pooling_tensor(
                            default_feature_result) if pooling == 'max' else average_pooling_tensor(default_feature_result),
                        'img_path': paths[i],
                        'state': 'dbox'
                    }

                    list_names[c].append(class_data)

            batch_end = time.time() - batch_time
            print(" Inference time for a image : {}".format(batch_end / batch_size))
            print(" Inference time for batch image : {}".format(batch_end))

            if rp:
                print("Reduce Vector Dimension 1024 to 512 ..")
                time_rd = time.time()
                rng = np.random.RandomState(42)
                transformer = SparseRandomProjection(n_components=512, random_state=rng)

                for i, ln in enumerate(list_names):
                    if list_names[i] != []:
                        ln_df = pd.DataFrame(pd.DataFrame.from_dict(ln)['feature_vector'].tolist())
                        ln_df_x = ln_df.loc[:, :].values  # numpy arrays
                        X_new = transformer.fit_transform(ln_df_x)
                        # 재저장
                        for j in range(len(ln)):
                            ln[j]['feature_vector'] = X_new[j]
                print(".. ", time.time() - time_rd)

        return list_names


    def vector_extraction_service(self, base_img, pooling='max'):
        img_res = {}
        list_names = []

        names = load_classes(self.names_path)

        for i in range(len(names)):
            list_names.append([])

        batch_time = time.time()
        torch.cuda.empty_cache()

        with torch.no_grad():
            img_bytes = base64.b64decode(base_img)
            file_bytes = np.asarray(bytearray(BytesIO(img_bytes).read()), dtype=np.uint8)
            decode_img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            img, img0 = transfer_b64(decode_img, mode='square')  # auto, square, rect, scaleFill / default : square
            #img.shpae: [3,416,416), img0.shape:[h, w, 3]
            img = torch.from_numpy(img).to(self.device)
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            layerResult = LayerResult(self.model.module_list, 80)

            pred = self.model(img)[0]

            layerResult_tensor = layerResult.features.permute([0, 2, 3, 1])
            kernel_size = layerResult_tensor.shape[1]

            LayerResult.unregister_forward_hook(layerResult)

            # box info
            pred = non_max_suppression(pred, conf_thres=0.5, nms_thres=0.5)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if det is not None and len(det):

                ratio = kernel_size / self.img_size
                resized_det = det[:, :4] * ratio

                im0shape = img0.shape
                feature_box = np.asarray(resized_det.detach().cpu().numpy(), dtype=np.int32)

                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0shape).round()  # originial

                for (*xyxy, conf, cls), fb in zip(det, feature_box):
                    #if conf < 0.5: continue
                    feature_result = layerResult_tensor[i, fb[0]:fb[2] + 1, fb[1]:fb[3] + 1]

                    xyxy = [int(x) for x in xyxy]
                    xyxy[0], xyxy[2] = xyxy[0] / im0shape[1], xyxy[2] / im0shape[1]
                    xyxy[1], xyxy[3] = xyxy[1] / im0shape[0], xyxy[3] / im0shape[0]

                    class_data = {
                        'raw_box': xyxy,
                        'feature_vector': max_pooling_tensor(
                            feature_result) if pooling == 'max' else average_pooling_tensor(feature_result),
                        'category' : names[int(cls.cpu().numpy())]
                    }

                    list_names[int(cls)].append(class_data)

        for i in range(len(names)):
            if list_names[int(i)] != []:
                img_res[names[int(i)]] = list_names[int(i)]

        batch_end = time.time() - batch_time
        print("Inference time for a image : {}".format(batch_end))

        return img_res

    def vector_extraction_service_testset(self, base_img, img_path):
        img_res = {}
        list_names = []

        names = load_classes(self.names_path)

        for i in range(len(names)):
            list_names.append([])

        batch_time = time.time()
        torch.cuda.empty_cache()

        with torch.no_grad():
            img_bytes = base64.b64decode(base_img)
            file_bytes = np.asarray(bytearray(BytesIO(img_bytes).read()), dtype=np.uint8)
            decode_img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            img, img0 = transfer_b64(decode_img, mode='square')  # auto, square, rect, scaleFill / default : square
            img = torch.from_numpy(img).to(self.device)
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            layerResult = LayerResult(self.model.module_list, 80)

            pred = self.model(img)[0]

            layerResult_tensor = layerResult.features.permute([0, 2, 3, 1])
            kernel_size = layerResult_tensor.shape[1]

            LayerResult.unregister_forward_hook(layerResult)

            # box info
            pred = non_max_suppression(pred, conf_thres=0.5, nms_thres=0.5)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if det is not None and len(det):

                ratio = kernel_size / self.img_size
                resized_det = det[:, :4] * ratio

                im0shape = img0.shape
                feature_box = np.asarray(resized_det.detach().cpu().numpy(), dtype=np.int32)

                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0shape).round()  # originial

                for (*xyxy, conf, cls), fb in zip(det, feature_box):
                    if conf < 0.5: continue
                    feature_result = layerResult_tensor[i, fb[0]:fb[2] + 1, fb[1]:fb[3] + 1]

                    xyxy = [int(x) for x in xyxy]

                    RESNET = Resnet.get_feature_vector(Resnet, xyxy, decode_img, svc=True)
                    MAC = max_pooling_tensor(feature_result)
                    SPoC = average_pooling_tensor(feature_result)
                    CAT = torch.cat((MAC, SPoC))

                    xyxy[0], xyxy[2] = xyxy[0] / im0shape[1], xyxy[2] / im0shape[1]
                    xyxy[1], xyxy[3] = xyxy[1] / im0shape[0], xyxy[3] / im0shape[0]

                    class_data = {
                        'raw_box': xyxy,
                        'yolo_vector_mac': MAC.detach().cpu().numpy(),
                        'yolo_vector_spoc': SPoC.detach().cpu().numpy(),
                        'yolo_vector_cat': CAT.detach().cpu().numpy(),
                        'resnet_vector': RESNET,
                        'category': mapping_ops[names[int(cls.cpu().numpy())]],
                        'img_size': im0shape,
                        'img_path': img_path
                    }

                    list_names[int(cls)].append(class_data)

        for i in range(len(names)):
            if list_names[int(i)] != []:
                img_res[names[int(i)]] = list_names[int(i)]

        batch_end = time.time() - batch_time
        print("Inference time for a image : {}".format(batch_end))

        return img_res


    def vector_extraction_service_full_backup(self, base_img, pooling='max'):
        list_names = []

        names = load_classes(self.names_path)

        for i in range(len(names)):
            list_names.append([])

        batch_time = time.time()
        torch.cuda.empty_cache()

        with torch.no_grad():
            img_bytes = base64.b64decode(base_img)
            file_bytes = np.asarray(bytearray(BytesIO(img_bytes).read()), dtype=np.uint8)
            decode_img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            img, img0 = transfer_b64(decode_img, mode='square')  # auto, square, rect, scaleFill / default : square
            #img.shpae: [3,416,416), img0.shape:[h, w, 3]
            img = torch.from_numpy(img).to(self.device)
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            layerResult = LayerResult(self.model.module_list, 80)

            pred = self.model(img)[0]

            layerResult_tensor = layerResult.features.permute([0, 2, 3, 1])
            kernel_size = layerResult_tensor.shape[1]

            LayerResult.unregister_forward_hook(layerResult)

            # box info
            pred = non_max_suppression(pred, conf_thres=0.5, nms_thres=0.5)

            for i in range(layerResult_tensor.shape[0]):
                MAC = max_pooling_tensor(layerResult_tensor[i])
                SPoC = average_pooling_tensor(layerResult_tensor[i])
                CAT = torch.cat((MAC, SPoC))

                data = {
                    'feature_vector_mac': MAC.detach().cpu().numpy(),
                    'feature_vector_spoc': SPoC.detach().cpu().numpy(),
                    'feature_vector_cat': CAT.detach().cpu().numpy(),
                }

        return data

    
    def vector_extraction_batch_full(self, bulk_path):
        
        vec = Cgd.get_cgd_vec(Cgd, bulk_path)

        result_list = []
        for i in range(vec.size()[1]):
            data = {
                'cgd_vector' : vec[:, i],
                'img_path': bulk_path[i],
                'state': 'full'
            }

            result_list.append(data)

        return result_list

    def vector_extraction_service_full(self, imagePIL):

        """ img_bytes = base64.b64decode(base_img)
        file_bytes = np.asarray(bytearray(BytesIO(img_bytes).read()), dtype=np.uint8)
        decode_img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        """
        vec_flat = Cgd.get_cgd_vec_decodeImg(Cgd, imagePIL)


        data = {
            'cgd_vector': vec_flat
        }

        return data
    

    def vector_extractor_by_model_V2(self, data, category, batch_size):
        names = load_classes(self.names_path)

        list_names = {}

        dataset = LoadImages(data, self.img_size, batch_size=batch_size)
        batch_size = min(batch_size, len(dataset))
        dataloader = DataLoader(dataset,
                                batch_size=batch_size,
                                num_workers=min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8]),
                                #num_workers=1,
                                pin_memory=True,
                                collate_fn=dataset.collate_fn)

        # [batch, 3, 416, 416], [path1, path2], [[h,w,3],[h,w,3]]
        for batch_i, (imgs, paths, shapes) in enumerate(dataloader):
            cgd_result = []
            torch.cuda.empty_cache()

            with torch.no_grad():
                imgs = imgs.to(self.device).float() / 255.0
                _, _, height, width = imgs.shape

                pred = self.model(imgs)[0]
                pred = non_max_suppression(pred, conf_thres=0.5, nms_thres=0.6)

            cgd_xyxy, cgd_xyxy_ori, cgd_img = [], [], []
            # Process detections
            for i, det in enumerate(pred):  # detections per image
                c = category[batch_i * batch_size + i]
                check = False
                try:
                    state = list_names[c].copy()
                except:
                    list_names[c] = []
                    state = list_names[c].copy()

                im0shape = shapes[i]  # original shape

                if det is not None and len(det):
                    det = torch.clamp(det, min=0, max=416)

                    det[:, :4] = scale_coords(imgs.shape[2:], det[:, :4], im0shape).round()  # originial

                    for *xyxy, conf, cls in det:

                        # 직선형태로 박스가 잡혔다면,
                        if conf < 0.3: continue

                        xyxy = [np.asarray(torch.round(x).cpu().numpy(), dtype=np.int32) for x in xyxy]
                        cgd_xyxy.append([int(x) for x in xyxy])
                        cgd_img.append(paths[i])

                        cgd_xyxy_ori.append([xyxy[0] / im0shape[1],xyxy[1] / im0shape[0], xyxy[2] / im0shape[1], xyxy[3] / im0shape[0]])
                        check = True

                #if list_names[c] == state:
                if not check:
                    data = defaultbox(im0shape, r=0.1)

                    xyxy = [round(d) for d in list(data[0]) + list(data[1])]

                    cgd_xyxy.append([int(x) for x in xyxy])
                    cgd_img.append(paths[i])

                    cgd_xyxy_ori.append([xyxy[0] / im0shape[1],xyxy[1] / im0shape[0], xyxy[2] / im0shape[1], xyxy[3] / im0shape[0]])

                    

            #CGD_62 = Cgd.get_cgd_vec_crop_batch(Cgd, cgd_xyxy, cgd_img, 62)
            #CGD_48 = Cgd.get_cgd_vec_crop_batch(Cgd, cgd_xyxy, cgd_img, 48)
            #CGD_10 = Cgd.get_cgd_vec_crop_batch(Cgd, cgd_xyxy, cgd_img, 10)
            #CGD_44 = Cgd.get_cgd_vec_crop_batch(Cgd, cgd_xyxy, cgd_img, 44)
            #CGD_90 = Cgd.get_cgd_vec_crop_batch(Cgd, cgd_xyxy, cgd_img, 90)
            CGD_7 = Cgd.get_cgd_vec_crop_batch(Cgd, cgd_xyxy, cgd_img, 7)
            
            for idx, path in enumerate(cgd_img):
                

                class_data = {
                    'raw_box': cgd_xyxy_ori[idx],
                    #'cgd62_vector' : CGD_62[idx],
                    #'cgd48_vector' : CGD_48[idx],
                    #'cgd10_vector' : CGD_10[idx],
                    #'cgd44_vector' : CGD_44[idx],
                    'cgd7_vector' : CGD_7[idx],
                    'img_path': path,
                    'state': 'fbox'
                }

                list_names[c].append(class_data)

        return list_names

