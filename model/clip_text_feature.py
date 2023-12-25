import os
import clip
import torch
from torchvision import datasets

class CLIP():
    def __init__(self):
        super().__init__()
        # self.device = device
        self.prompt = "Detect: {}."
    
    def load_device(self, device):
        self.device = device
        self.model, _= clip.load('ViT-B/32', self.device)

    def text_feature(self, object_titles):
        
        res = []

        for object_title in object_titles:

            text_inputs =  (clip.tokenize(self.prompt.format(object_title))).to(self.device)

            with torch.no_grad():
                text_features = self.model.encode_text(text_inputs)
                text_features /= text_features.norm(dim=-1, keepdim=True)
            
            res.append(text_features)
        
        res = torch.cat(res, dim=0)
            

        return res
        