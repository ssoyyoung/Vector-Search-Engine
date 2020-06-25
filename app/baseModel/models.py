import torch
import torchvision.models as models

class Resnet18():
    def __init__(self):
        self.model = models.resnet18(pretrained=True)
        self.FE = torch.nn.Sequential(*list(self.model.children())[:-1])
        print("[RESNET18] Loading Resnet18 vector extractor Model...")