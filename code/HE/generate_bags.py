from network import LUCCModel
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import glob
import h5py

maxTileNum=500
bag_path = '../../data/cancer/jpg'
model_path = 'checkpoint_500.tar'
model = LUCCModel(class_num=2)
model.load_state_dict(torch.load(model_path)['net'])
model = model.cuda()
model.eval()
device = torch.device('cuda')
bags = glob.glob(bag_path+'/*')
with torch.no_grad():
    for bag in bags:
        bag_name = bag.split('\\')[-1]
        instances = glob.glob(bag+'/*.jpeg')[:50]
        if len(instances) > maxTileNum:
            instances = np.random.choice(instances,maxTileNum)
        features = None
        for instance in instances:
            img = torch.from_numpy(np.array(Image.open(instance).convert('RGB'))).float() / 255.0
            img = img.to(device)
            img = img.permute(2,0,1)
            img = img.unsqueeze(0)
            feature = model.feature(img)
            if features is None:
                features = feature
                zero_tensor = torch.zeros_like(feature)
            else:
                features = torch.cat((features,feature),dim=0)
        if len(instances) < maxTileNum:
            for _ in range(maxTileNum-len(instances)):
                features = torch.cat((features,zero_tensor),dim=0)
        print(bag_name,features.shape,len(instances) )
        features = features.detach().cpu().numpy()
        with h5py.File(bag_name+'.hdf5','w') as f:
            f.create_dataset('feature',data = features)
            f.create_dataset('length',data = len(instances))
            f.create_dataset('label',data = bag_name)

