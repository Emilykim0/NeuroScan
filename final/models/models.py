# models.py
import torch
import torch.nn as nn
import random
import numpy as np
import torch

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


class LConvNet(nn.Module):
    def __init__(self, num_channels=12, seq_len=256, num_classes=2):
        super(LConvNet, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(3,3), padding=(1,1)),
            nn.ReLU(),
            nn.MaxPool2d((1,2)),

            nn.Conv2d(16, 32, kernel_size=(3,3), padding=(1,1)),
            nn.ReLU(),
            nn.MaxPool2d((1,2)),

            nn.Conv2d(32, 64, kernel_size=(3,3), padding=(1,1)),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.MaxPool2d((1,2)),

            nn.AdaptiveAvgPool2d((12, 16))  # (B, 64, 12, 16)
        )

        self.flatten = nn.Flatten(start_dim=2)      # (B, 64, 192)
        self.time_dense = nn.Linear(64, 32)         # FIXED: (B, 192, 64) → (B, 192, 32)
        self.lstm = nn.LSTM(32, 32, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(32, 25),
            nn.ReLU(),
            nn.Linear(25, num_classes)
        )

    def forward(self, x):
        x = x.unsqueeze(1)          # (B, 1, 12, 256)
        x = self.cnn(x)             # (B, 64, 12, 16)
        x = self.flatten(x)         # (B, 64, 192)
        x = x.permute(0, 2, 1)      # (B, 192, 64)
        x = self.time_dense(x)      # (B, 192, 32)
        x, _ = self.lstm(x)         # (B, 192, 32)
        x = x[:, -1, :]             # (B, 32)
        out = self.classifier(x)    # (B, 2)
        return out

class CNN_LSTM_Hybrid_Final(nn.Module):
    def __init__(self, num_channels=12, seq_len=256, num_classes=2):
        super(CNN_LSTM_Hybrid_Final, self).__init__()

        # [1] CNN feature extractor (전이학습 기반 스타일)
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(num_channels, 32, kernel_size=7, padding=3),  # 큰 커널로 넓게 보기
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),  # 256 -> 128

            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),  # 128 -> 64

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # [2] LSTM temporal modeling
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )

        # [3] Classifier
        self.classifier = nn.Sequential(
            nn.Linear(128 * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # x shape: (batch, 12, 256)
        x = self.feature_extractor(x)  # (batch, 128, 64)
        x = x.permute(0, 2, 1)          # (batch, 64, 128)
        out, _ = self.lstm(x)           # (batch, 64, 256)
        out = out[:, -1, :]             # 마지막 timestep만 사용
        out = self.classifier(out)      # (batch, 2)
        return out
