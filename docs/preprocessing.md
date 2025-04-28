# 🧹 EEG 데이터 전처리 과정

## 1. 기본 흐름
- 0.5–40Hz Bandpass Filtering
- 128Hz로 Resampling
- 2초 슬라이딩 (stride=1초, 50% overlap)
- 주요 12개 채널 선택
- 채널별 Z-score 정규화

## 2. 최종 전처리 포맷
- 입력 데이터: (N, 12, 256)
- 레이블: (0: Normal, 1: Seizure)

## 3. 전처리 코드
- `/initial/preprocessing/`
- `/final/preprocessing/`
