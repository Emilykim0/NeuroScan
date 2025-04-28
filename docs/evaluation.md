# 📊 평가 지표 및 성능 요약

## 1. 주요 평가 지표
- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix (Normal / Seizure)

## 2. 최종 성능 비교

| 모델명 | Validation Accuracy | Test Accuracy | Test F1 Score |
|:---|:---|:---|:---|
| CNN_LSTM_Hybrid | 0.9088 | 0.8986 | 0.8991 |
| LConvNet | 0.8806 | 0.8596 | 0.8692 |
| CNN_LSTM_Hybrid_Final | 0.9031 | 0.9027 | 0.9050 |

✅ 최종 채택 모델 기준:
- Test Accuracy: **90.27%**
- Test F1 Score: **90.50%**

## 3. 평가 코드
- `/initial/train_evaluate/`
- `/final/train_evaluate/`
