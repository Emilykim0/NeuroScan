# 🏗️ 모델 구조 및 특징

## 1. 주요 모델 요약

| 모델 | 구조 요약 | 특징 |
|:---|:---|:---|
| CNN_LSTM3 | 2 Conv1D + 2 BiLSTM + FC | 초기 Best |
| CNN_LSTM_Hybrid | 전이학습 기반 CNN + BiLSTM | 초기 최고 성능 (Val Acc 0.8557) |
| CNN_LSTM_Hybrid_Final | 간결한 CNN + BiLSTM (최종 전처리 최적화) | 최종 Best (Test Acc 90.27%) |
| LConvNet | 2D CNN + LSTM 조합 (논문 기반) | Recall 우수 |

✅ 최종 채택 모델: **CNN_LSTM_Hybrid_Final**

## 2. 모델 코드
- `/initial/models/`
- `/final/models/`
