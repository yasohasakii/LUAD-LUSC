import network
import torch
import torch.nn as nn

model_path = 'checkpoint_500.tar'
model = network.LUCCModel(class_num=2)
model.load_state_dict(torch.load(model_path)['net'])
model.eval()

data = torch.randn(5,3,256,256)
out1 = model.feature(data)
m2 = network.MILModel(n_feats=128,n_out=1)
out2 = m2((out1.reshape((1,5,-1)),torch.Tensor(1)))
print(out2)