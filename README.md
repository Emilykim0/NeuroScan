# 🧠 NeuroScan: EEG Seizure Detection Project

## 📌 프로젝트 개요

- **주제:** EEG 신호 기반 발작(Seizure) 감지 모델 개발 및 웹 서비스 구축
- **모델 종류:** CNN-LSTM 계열, 전이학습 기반 하이브리드 모델
- **웹 링크:** 로컬 실행 (Streamlit 기반, `web/web_app.py`)

> 📄 프로젝트 문서 상세 보기:  
> 🔗 [EEG Seizure 1 - 데이터 구성 및 전처리, EDA 분석](https://yeonghyekim.notion.site/EEG_Seizure_1-1d1e2859370c8063a0f1ce15d28e0147?pvs=4)  
> 🔗 [EEG Seizure 2 - 모델 개발, 성능 평가 및 웹 서비스 구현](https://yeonghyekim.notion.site/EEG_Seizure_2-1e0e2859370c808a8b4ef7b2d84c6695?pvs=4)

---

## 📂 폴더 구조

```bash
NeuroScan/
│
├── initial/                # 초기 전처리 및 모델링 버전
│   ├── DATA/
│   ├── models/
│   ├── preprocessing/
│   ├── saved_models/
│   └── train_evaluate/
│
├── final/                  # 최종 전처리 및 모델링 버전
│   ├── DATA/
│   ├── models/
│   ├── preprocessing/
│   ├── saved_models/
│   └── train_evaluate/
│
├── web/                    # Streamlit 웹 구현 폴더
│   ├── saved_models/
│   ├── edf_to_npy.py
│   ├── sliding_predict.py
│   ├── seizure_utils.py
│   ├── models.py
│   ├── web_app.py
│   └── neuron.png
│
├── docs/                   # 프로젝트 문서 모음
│   ├── preprocessing.md
│   ├── models.md
│   ├── evaluation.md
│   └── web.md
│
├── RECORDS-WITH-SEIZURES    # 발작 발생 기록 파일
└── README.md                # (현재 파일)
```
---

# 🔍 데이터 개요

- 0.5–40Hz Bandpass Filtering
- 128Hz Resampling
- 2초 단위 슬라이딩 (256포인트)
- 채널 수: 12개 유지
- 채널별 Z-score 정규화
- 최종 입력 형태: **(N, 12, 256)**

---

# 🧠 모델 요약

| 모델 | 구조 요약 | 특징 |
|:---|:---|:---|
| CNN_LSTM3 | 2 Conv1D + 2 BiLSTM + FC | 초기 Best |
| CNN_LSTM_Hybrid | 전이학습 기반 CNN + BiLSTM | 초기 최고 성능 (Val Acc 0.8557) |
| CNN_LSTM_Hybrid_Final | 간결한 CNN + BiLSTM (최종 전처리 최적화) | 최종 Best (Test Acc 90.27%) |

✅ 최종 채택 모델: **CNN_LSTM_Hybrid_Final**

---

# 📈 모델 성능 요약

| 모델명 | Validation Accuracy | Test Accuracy | Test F1 Score |
|:---|:---|:---|:---|
| CNN_LSTM_Hybrid | 0.9088 | 0.8986 | 0.8991 |
| CNN_LSTM_Hybrid_Final | 0.9031 | 0.9027 | 0.9050 |

---

# ✅ 웹 서비스 요약

- EEG .edf 파일 업로드 → 발작 구간 자동 분석
- 발작 episode 단위로 결과 제공
- 대화형 Plotly 시각화
- 총 발작 횟수 및 총 발작 시간 리포트

```bash
cd web
streamlit run web_app.py
```

---

# 📌 문서 상세 보기

- Preprocessing.md → 데이터 전처리 흐름
- Models.md → 모델 구조 및 설계
- Evaluation.md → 평가 지표 및 성능 요약
- Web.md → 웹 구조 및 기능 설명

---

# 👩‍💻 제작자

- **이름:** Emily
- **역할:** 전처리, 모델링, 웹 구현, 문서 작성


---

## 📂 `docs/` 폴더 안 파일들

### 1️⃣ preprocessing.md

```markdown
# 🧹 EEG 데이터 전처리 과정

## 1. 기본 흐름
- Bandpass Filtering (0.5–40Hz)
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
```

---

### 2️⃣ models.md

```markdown
# 🏗️ 모델 구조 및 특징

## 1. 주요 모델 요약

| 모델 | 구조 요약 | 특징 |
|:---|:---|:---|
| CNN_LSTM3 | 2 Conv1D + 2 BiLSTM + FC | 초기 Best |
| CNN_LSTM_Hybrid | 전이학습 기반 CNN + BiLSTM | 초기 최고 성능 (Val Acc 0.8557) |
| CNN_LSTM_Hybrid_Final | 간결한 CNN + BiLSTM (최종 전처리 최적화) | 최종 Best (Test Acc 90.27%) |

✅ 최종 채택 모델: **CNN_LSTM_Hybrid_Final**

## 2. 모델 코드
- `/initial/models/`
- `/final/models/`
```

---

### 3️⃣ evaluation.md

```markdown
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
| CNN_LSTM_Hybrid_Final | 0.9031 | 0.9027 | 0.9050 |

✅ 최종 채택 모델 기준:
- Test Accuracy: **90.27%**
- Test F1 Score: **90.50%**

## 3. 평가 코드
- `/initial/train_evaluate/`
- `/final/train_evaluate/`
```

---

### 4️⃣ web.md

```markdown
# 🌐 웹 구조 및 기능 설명

## 1. 웹 기능 요약
- EEG `.edf` 파일 업로드
- EEG 신호 전처리 및 세그먼트화
- 발작 구간 실시간 예측
- 발작 episode 병합 및 리포트
- 대화형 시각화 (Plotly 사용)

## 2. Streamlit 메인 파일
- `web/web_app.py`

## 3. 주요 모듈
| 파일 | 역할 |
|:---|:---|
| edf_to_npy.py | EDF 파일 → 2초 세그먼트 변환 |
| sliding_predict.py | 세그먼트 입력 → 발작 여부 예측 |
| seizure_utils.py | 발작 episode 분석 및 시각화 |
| models.py | CNN_LSTM_Hybrid_Final 구조 포함 |

## 4. 실행 방법

```bash
cd web
streamlit run web_app.py
```

