import torch
from torchvision import transforms
from utils.vector_extractor_v2 import Yolov3
from PIL import Image
import math

class Resnet:

    model_path = "resnet/resnet_irs_v5"
    model = torch.load(model_path)
    model.eval()

    def resnet_vector_service(self, img_path, cate):

        box = self.get_rawbox(self,[img_path],[cate])

        for idx in box.keys():
            input_image = Image.open(box[idx]['img_path'])
            width, height = input_image.size
            if box[idx]['state'] == 'dbox':
                xy = [0.1, 0.1, 0.9, 0.9]
            elif box[idx]['state'] == 'fbox':
                xy = box[idx]['raw_box']
            left = math.floor(width * (float(xy[0])))
            top = math.floor(height * (float(xy[1])))
            right = math.ceil(width * (float(xy[2])))
            bottom = math.ceil(height * (float(xy[3])))

            crop_image = input_image.crop((left, top, right, bottom))

            input_size = 224
            preprocess = transforms.Compose([
                transforms.Resize(input_size),
                transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            input_tensor = preprocess(crop_image)
            input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

            if torch.cuda.is_available():
                input_batch = input_batch.to('cuda')
                self.model.to('cuda')

            with torch.no_grad():
                self.model(input_batch)

            fv = (self.model.vec_out[0].cpu().numpy()).tolist()
            box[idx]['feature_vector'] = fv

        return box



