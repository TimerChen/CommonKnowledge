import torch
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import numpy as np


def default_loader(path):
    return Image.open(path).convert('1')

class MyDataset(Dataset):
    def __init__(self, txt_path, pic_path, transform=None, target_transform=None, loader=default_loader):
#        fh = open(txt, 'r')
#        data = pd.DataFrame()
        data = pd.read_csv(txt_path)
        data = np.array(data)
        imgs = []
        for i in range(data.shape[0]):
            imgs.append(i)
        self.data = data
        self.pic_path = pic_path
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        label = self.data[index, 1]
        img = self.loader(self.pic_path +'/'+ str(index) + ".png")
        if self.transform is not None:
            img = self.transform(img)
        print(type(img), type(label))
        return img, label

    def __len__(self):
        return len(self.imgs)
