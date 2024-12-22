import os
import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

class CIFAR10Dataset(Dataset):
    def __init__(self, data_dir, train=True, transform=None):
        self.data_dir = data_dir
        self.train = train
        self.transform = transform
        self.data, self.labels = self.load_data()
    
    def load_data(self):
        if self.train:
            data_batches = [self.unpickle(os.path.join(self.data_dir, f'data_batch_{i}')) for i in range(1, 6)]
            data = np.concatenate([batch[b'data'] for batch in data_batches])
            labels = np.concatenate([batch[b'labels'] for batch in data_batches])
        else:
            data_dict = self.unpickle(os.path.join(self.data_dir, 'test_batch'))
            data = data_dict[b'data']
            labels = data_dict[b'labels']
        
        data = data.reshape(-1, 3, 32, 32).astype("float32")
        labels = np.array(labels)
        return data, labels
    
    def unpickle(self, file):
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image = self.data[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

def get_dataloader(data_dir, batch_size, train=True):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    dataset = CIFAR10Dataset(data_dir, train=train, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=train)
    return dataloader
