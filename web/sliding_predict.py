import torch
import numpy as np
from models import CNN_LSTM_Hybrid_Final

# ------------------ 설정 ------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "./saved_models/CNN_LSTM_Hybrid_Final_acc_0.9031.pt"

# 모델 로드
model = CNN_LSTM_Hybrid_Final(num_channels=12, seq_len=256, num_classes=2).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE,weights_only=True))
model.eval()

# ------------------ 예측 함수 ------------------
def predict_seizure_from_segments(segments_np):
    """
    segments_np: shape (N, 12, 256) - 이미 2초 단위로 슬라이딩된 npy 파일
    """
    X_tensor = torch.tensor(segments_np, dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        outputs = model(X_tensor)
        probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
        preds = (probs > 0.8).astype(int)  # 확률이 0.8 넘을 때만 발작으로 간주
    return preds, probs
