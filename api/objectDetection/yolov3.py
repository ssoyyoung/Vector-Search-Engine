import pandas as pd
import json
from sklearn.random_projection import SparseRandomProjection
from api.objectDetection.resnet import Resnet

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


class YOLOV3:
    print("Loading Yolov3 model ... ")

    img_size = 416
    cfg = 'yolov3/cfg/fashion/fashion_c23.cfg'
    weights = '/mnt/piclick/piclick.ai/weights/best.pt'
    names_path = 'yolov3/data/fashion/fashion_c23.names'

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

                        RESNET = Resnet.get_feature_vector(Resnet, [int(x) for x in xyxy], paths[i])
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
                            'img_path': paths[i],
                            'state': 'fbox'
                        }

                        list_names[c].append(class_data)


                # if no change
                if list_names[c] == state:
                    default_feature_result = layerResult_tensor[i, 1:12, 1:12]
                    data = defaultbox(im0shape, r=0.1)

                    xyxy = [round(d) for d in list(data[0]) + list(data[1])]

                    RESNET = Resnet.get_feature_vector(Resnet, xyxy, paths[i])
                    MAC = max_pooling_tensor(default_feature_result)
                    SPoC = average_pooling_tensor(default_feature_result)
                    SPoc_wo_norm = average_pooling_tensor(default_feature_result, False)
                    CAT = torch.cat((MAC, SPoC))

                    xyxy[0], xyxy[2] = xyxy[0] / im0shape[1], xyxy[2] / im0shape[1]
                    xyxy[1], xyxy[3] = xyxy[1] / im0shape[0], xyxy[3] / im0shape[0]

                    class_data = {
                        'raw_box': xyxy,
                        'yolo_vector_mac': MAC.detach().cpu().numpy(),
                        'yolo_vector_spoc': SPoC.detach().cpu().numpy(),
                        'yolo_vector_spoc_w/o_norm': SPoc_wo_norm.detach().cpu().numpy(),
                        'yolo_vector_cat': CAT.detach().cpu().numpy(),
                        'resnet_vector': RESNET,
                        'img_path': paths[i],
                        'state': 'fbox'
                    }

                    list_names[c].append(class_data)

            #print(" Inference time for batch image : {}".format(batch_end))

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