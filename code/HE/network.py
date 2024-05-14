import timm
import torch
import torch.nn as nn
from torch.nn.functional import normalize

a = torch.rand((5,3,256,256))

def get_model_output_size(model):
    names = list(model._modules.keys())
    for idx in range(len(names), 0, -1):
        idx -= 1
        if not isinstance(model._modules[names[idx]],nn.modules.activation.ReLU):
            break
    module = model._modules[names[idx]]
    if 'weight' not in dir(module):
        return get_model_output_size(module)
    size = len(module.weight)
    return size

'''
    for idx in range(len(names),0,-1):
        idx-=1
        if 'weight' in dir(model._modules[names[idx]]):
            break
        else:
            print(names[idx],' has none weight.')
'''


class LUCCModel(nn.Module):
    """
    Model for Lung cancer contrastive clustering.
    """
    def __init__(self,basemodel = 'resnet34'):
        super(LUCCModel,self).__init__()
        self.backbone = timm.create_model(basemodel,pretrained=True)
        size = get_model_output_size(self.backbone)
        # define instance-level contrastrive head
        self.ILCH = nn.Sequential(nn.Linear(size,size),
                                  nn.ReLU(),
                                  nn.Linear(size,size))
        # define cluster-level contrastrive head
        self.CLCH = nn.Sequential(nn.Linear(size,size),
                                  nn.ReLU(),
                                  nn.Linear(size,size),
                                  nn.Softmax(dim=1))
        self.adaptive_pool = nn.AdaptiveAvgPool2d((224,224))

    def forward(self,x1,x2):
        x1 = self.adaptive_pool(x1)
        x2 = self.adaptive_pool(x2)
        x1 = self.backbone(x1)
        x2 = self.backbone(x2)
        z_1 = normalize(self.ILCH(x1),dim=1)
        z_2 = normalize(self.ILCH(x2),dim=1)
        c_1 = self.CLCH(x1)
        c_2 = self.CLCH(x2)
        return z_1,z_2,c_1,c_2

    def cluster(self,x):
        x = self.adaptive_pool(x)
        c = self.CLCH(x)
        c = torch.argmax(c,dim=1)
        return c
model = LUCCModel()
o1,o2,o3,o4 = model(a,a)
print(o1.shape,o2.shape,o3.shape,o4.shape)