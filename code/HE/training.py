import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

a = torch.rand((1,3,256,256))
class CancerModel(nn.Module):
    def __init__(self):
        super(CancerModel,self).__init__()
        self.backbone = timm.create_model('resnet34',pretrained=True,num_classes = 3)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((224,224))

    def forward(self,x):
        x = self.adaptive_pool(x)
        x = self.backbone(x)
        x = F.relu(x,inplace=True)
        return x
model = CancerModel()
o = model(a)
print(o,o.shape)
