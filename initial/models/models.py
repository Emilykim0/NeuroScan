import torch
import torch.nn as nn
import random
import numpy as np
import torch

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

class EEG_LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, num_classes=2, dropout=0.3):
        super(EEG_LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True,
                            dropout=dropout)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x: (batch, seq_len, 1)
        out, _ = self.lstm(x)  # out: (batch, seq_len, hidden)
        out = out[:, -1, :]    # 마지막 시점의 hidden만 사용
        out = self.fc(out)     # (batch, num_classes)
        return out



class EEG_LSTM_Deep(nn.Module):
    def __init__(self, input_size=1, hidden_size=128, num_layers=2, num_classes=2, dropout=0.3):
        super(EEG_LSTM_Deep, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True,
                            dropout=dropout,
                            bidirectional=True)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 128),  # bidirectional → hidden*2
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # 마지막 시점의 출력만
        out = self.classifier(out)
        return out
    
#=================================================================================
# 다채널: (batch, channels, seq_len) → (batch, seq_len, channels)

class EEG_LSTM_Multi(nn.Module):
    def __init__(self, input_size=12, hidden_size=64, num_layers=2, num_classes=2, dropout=0.3):
        super(EEG_LSTM_Multi, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x: (batch, channels=12, time=256) → (batch, time, channels)
        x = x.permute(0, 2, 1)
        out, _ = self.lstm(x)              # (batch, time, hidden)
        out = out[:, -1, :]                # 마지막 타임스텝만
        out = self.fc(out)                 # (batch, num_classes)
        return out
    
#========================================================================================
# 다채널 & CNN-LSTM 
class CNN_LSTM(nn.Module):
    def __init__(self, num_channels=12, seq_len=256, num_classes=2):
        super(CNN_LSTM, self).__init__()

        # CNN 부분 (Channel-wise feature extraction)
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=num_channels, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),     # 256 -> 128

            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)      # 128 -> 64
        )

        # LSTM 부분 (Temporal modeling)
        self.lstm = nn.LSTM(input_size=64, hidden_size=128, num_layers=1, batch_first=True)

        # FC 부분 (Classification)
        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # x: (batch, channels=12, seq_len=256)
        x = self.cnn(x)                 # -> (batch, 64, 64)
        x = x.permute(0, 2, 1)          # -> (batch, 64, 64)
        out, _ = self.lstm(x)           # -> (batch, 64, 128)
        out = out[:, -1, :]             # 마지막 timestep만 추출 -> (batch, 128)
        out = self.fc(out)              # -> (batch, num_classes)
        return out




class CNN_LSTM2(nn.Module):
    def __init__(self, num_channels=12, seq_len=256, num_classes=2):
        super(CNN_LSTM2, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv1d(num_channels, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),  # 256 → 128

            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),  # 128 → 64
        )

        self.lstm = nn.LSTM(input_size=64, hidden_size=128, num_layers=1, batch_first=True)

        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.cnn(x)              # (B, 64, 64)
        x = x.permute(0, 2, 1)       # (B, 64, 64)
        out, _ = self.lstm(x)        # (B, 64, 128)
        out = out[:, -1, :]          # 마지막 timestep만 사용
        out = self.fc(out)           # (B, num_classes)
        return out


class CNN_LSTM3(nn.Module):
    def __init__(self, num_channels=12, seq_len=256, num_classes=2):
        super(CNN_LSTM3, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv1d(num_channels, 64, kernel_size=5, padding=2),   # Conv1: 더 깊고 넓은 필터
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),  # 256 → 128

            nn.Conv1d(64, 128, kernel_size=3, padding=1),             # Conv2
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),  # 128 → 64

            nn.Dropout(0.3)
        )

        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )

        self.fc = nn.Sequential(
            nn.Linear(128 * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.cnn(x)          # → (batch, 128, 64)
        x = x.permute(0, 2, 1)   # → (batch, 64, 128)
        out, _ = self.lstm(x)    # → (batch, 64, 256)
        out = out[:, -1, :]      # 마지막 timestep
        out = self.fc(out)       # → (batch, 2)
        return out




class CNN_LSTM4(nn.Module):
    def __init__(self, num_channels=12, seq_len=256, num_classes=2):
        super(CNN_LSTM4, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv1d(num_channels, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),  # 256 → 128

            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),  # 128 → 64

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)   # 64 → 32
        )

        self.lstm = nn.LSTM(input_size=128, hidden_size=128,
                            num_layers=2, batch_first=True,
                            bidirectional=True)

        self.fc = nn.Sequential(
            nn.Linear(128 * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.cnn(x)               # (batch, 128, 32)
        x = x.permute(0, 2, 1)        # (batch, seq_len=32, features=128)
        out, _ = self.lstm(x)         # (batch, 32, 256)
        out = out[:, -1, :]           # 마지막 time step
        out = self.fc(out)            # (batch, num_classes)
        return out

#============================================
# 전이학습
import torch
import torch.nn as nn

class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, use_bn=True):
        super().__init__()
        padding = kernel_size // 2

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, bias=not use_bn)
        self.bn1 = nn.BatchNorm1d(out_channels) if use_bn else nn.Identity()
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, 1, padding, bias=not use_bn)
        self.bn2 = nn.BatchNorm1d(out_channels) if use_bn else nn.Identity()

        self.downsample = nn.Sequential()
        if in_channels != out_channels or stride != 1:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels) if use_bn else nn.Identity()
            )

    def forward(self, x):
        identity = self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)


class ResNet1D_EEG(nn.Module):
    def __init__(self,
                 in_channels=12,
                 base_filters=64,
                 kernel_size=7,
                 stride=2,
                 n_block=4,
                 groups=1,
                 n_classes=2,
                 downsample_gap=2,
                 increasefilter_gap=2,
                 use_bn=True):
        super().__init__()
        self.in_channels = in_channels
        self.groups = groups

        self.first_block = nn.Sequential(
            nn.Conv1d(in_channels, base_filters, kernel_size=kernel_size, stride=stride,
                      padding=kernel_size // 2, bias=False),
            nn.BatchNorm1d(base_filters) if use_bn else nn.Identity(),
            nn.ReLU(inplace=True)
        )

        self.layers = nn.Sequential()
        filters = base_filters
        for i in range(n_block):
            downsample = (i % downsample_gap == 0)
            increase = (i % increasefilter_gap == 0)
            out_filters = filters * 2 if increase else filters
            stride = 2 if downsample else 1
            block = ResidualBlock1D(filters, out_filters, kernel_size, stride, use_bn)
            self.layers.add_module(f"resblock{i+1}", block)
            filters = out_filters

        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(filters, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, n_classes)
        )

    def forward(self, x):  # (batch, 12, 256)
        x = self.first_block(x)
        x = self.layers(x)
        x = self.global_pool(x)
        x = self.fc(x)
        return x
    
#------------------------------------------------------------------------------------------------------

import torch
import torch.nn as nn

class CNN_LSTM_Hybrid(nn.Module):
    def __init__(self, num_channels=12, seq_len=256, num_classes=2):
        super(CNN_LSTM_Hybrid, self).__init__()

        # 전이학습 기반 CNN 블록 (얕지만 일반화된 특징 추출용)
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(num_channels, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),  # 256 → 128

            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),  # 128 → 64

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )

        self.classifier = nn.Sequential(
            nn.Linear(128 * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.feature_extractor(x)  # → (B, 128, 64)
        x = x.permute(0, 2, 1)         # → (B, 64, 128)
        out, _ = self.lstm(x)          # → (B, 64, 256)
        out = out[:, -1, :]            # 마지막 timestep만
        out = self.classifier(out)     # → (B, 2)
        return out

