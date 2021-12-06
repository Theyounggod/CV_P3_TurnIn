from torch.utils.data import Dataset
from torch.autograd import Variable
import torchvision.transforms as transforms
import torch.nn as nn
import numpy as np
import torch

from typing import List

import csv


class SignLanguageMNIST(Dataset):
   
    @staticmethod
    def get_Mapping():
       
        mapping = list(range(25))
        mapping.pop(9)
        return mapping

    @staticmethod
    def read_CSV(path: str):

        mapping = SignLanguageMNIST.get_Mapping()
        labels, samples = [], []
        with open(path) as f:
            _ = next(f)  # skip header
            for line in csv.reader(f):
                label = int(line[0])
                labels.append(mapping.index(label))
                samples.append(list(map(int, line[1:])))
        return labels, samples

    def __init__(self,
            path: str="C:/Users/tyran/Desktop/proj_3/data/sign_mnist_train.csv",
            mean: List[float]=[0.485],
            std: List[float]=[0.229]):
        
        
        labels, samples = SignLanguageMNIST.read_CSV(path)
        self._samples = np.array(samples, dtype=np.uint8).reshape((-1, 28, 28, 1))
        self._labels = np.array(labels, dtype=np.uint8).reshape((-1, 1))

        self._mean = mean
        self._std = std

    def __len__(self):
        return len(self._labels)

    def __getitem__(self, idx):
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomResizedCrop(28, scale=(0.8, 1.2)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self._mean, std=self._std)])

        return {
            'image': transform(self._samples[idx]).float(),
            'label': torch.from_numpy(self._labels[idx]).float()
        }

def get_TLoader(batch_size=32):
    trainset = SignLanguageMNIST('C:/Users/tyran/Desktop/proj_3/data/sign_mnist_train.csv')
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

    testset = SignLanguageMNIST('C:/Users/tyran/Desktop/proj_3/data/sign_mnist_test.csv')
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)
    return trainloader, testloader

if __name__ == '__main__':
    loader, _ = get_TLoader(2)
    print(next(iter(loader)))