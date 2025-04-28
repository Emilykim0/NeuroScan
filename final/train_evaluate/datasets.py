# dataset.py
import numpy as np
import torch
from torch.utils.data import Dataset

class EEGDataset(torch.utils.data.Dataset):
    def __init__(self, data_x, data_y):
        # npy 파일 경로일 수도 있고, ndarray일 수도 있음
        if isinstance(data_x, str):
            self.data = np.load(data_x)
        else:
            self.data = data_x

        if isinstance(data_y, str):
            self.labels = np.load(data_y)
        else:
            self.labels = data_y

        # 데이터 타입 변환
        self.data = torch.tensor(self.data, dtype=torch.float32)
        self.labels = torch.tensor(self.labels, dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]