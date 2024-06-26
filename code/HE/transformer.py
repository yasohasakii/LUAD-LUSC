import glob
import os
import torchvision
import cv2
from torch.utils.data import Dataset
import numpy as np
from PIL import Image


class GaussianBlur:
    def __init__(self, kernel_size, min=0.1, max=2.0):
        self.min = min
        self.max = max
        self.kernel_size = kernel_size

    def __call__(self, sample):
        sample = np.array(sample)
        prob = np.random.random_sample()
        if prob < 0.5:
            sigma = (self.max - self.min) * np.random.random_sample() + self.min
            sample = cv2.GaussianBlur(sample, (self.kernel_size, self.kernel_size), sigma)
        return sample

class CIDataset(Dataset):
    def __init__(self,root,transform = None):
        self.transform = transform
        self.data = glob.glob(os.path.join(root,'*.png'))

    def __getitem__(self, item):
        path = self.data[item]
        img = cv2.cvtColor(cv2.imread(path),cv2.COLOR_BGR2RGB)
        if self.transform is not None:
            img = self.transform(img)
        if '/p/' in path:
            label = 1
        elif '/n/' in path:
            label = 0
        return img,label
    def __len__(self):
        return len(self.data)

class CLSDataset(Dataset):
    def __init__(self,root,transform = None):
        self.transform = transform
        self.data = glob.glob(os.path.join(root,'*.jpeg'))
        self.label_dict = {
            'colon_aca': 1,
            'colon_n': 0,
            'lung_aca': 2,
            'lung_n': 0,
            'lung_scc': 3,
        }
    def __getitem__(self, item):
        path = self.data[item]
        img = Image.open(path)#cv2.cvtColor(cv2.imread(path),cv2.COLOR_BGR2RGB)
        if self.transform is not None:
            img = self.transform(img)
        label = self.label_dict[path.split('\\')[-2]]
        return img,label
    def __len__(self):
        return len(self.data)



class Transforms:
    def __init__(self, s=1.0, mean=None, std=None, blur=False):
        self.train_transform = [
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomApply([torchvision.transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)],
                                               p=0.8),
            torchvision.transforms.RandomGrayscale(p=0.2),
        ]
        if blur:
            self.train_transform.append(GaussianBlur(kernel_size=23))
        self.train_transform.append(torchvision.transforms.ToTensor())
        self.test_transform = [
            torchvision.transforms.ToTensor(),
        ]
        if mean and std:
            self.train_transform.append(torchvision.transforms.Normalize(mean=mean, std=std))
            self.test_transform.append(torchvision.transforms.Normalize(mean=mean, std=std))
        self.train_transform = torchvision.transforms.Compose(self.train_transform)
        self.test_transform = torchvision.transforms.Compose(self.test_transform)

    def __call__(self, x):
        return self.train_transform(x), self.train_transform(x)