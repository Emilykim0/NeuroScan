import numpy as np
import torch
from torch.utils.data import Dataset
import random

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)


class EEGDatasetMulti(Dataset):
    def __init__(self, X_path, y_path, transform=None):
        self.X = np.load(X_path)        # (N, 12, 256)
        self.y = np.load(y_path)        # (N,)
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]  # numpy
        x = (x - x.mean(axis=1, keepdims=True)) / (x.std(axis=1, keepdims=True) + 1e-8) # (5) 정규화
        # x = x * 10  # 정규화 후 값 범위를 넓혀줘서 정보 보존
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(self.y[idx], dtype=torch.long)

        return x, y

    