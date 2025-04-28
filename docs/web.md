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

| 파일명 | 역할 |
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

## 5. 왜 episode 단위로 발작을 분석했는가?

- 개별 segment 예측은 노이즈나 일시적인 오류에 의해 쉽게 흔들릴 수 있다.
- 발작은 본질적으로 연속적인 뇌파 패턴 변화를 수반하므로, 지속성을 고려하는 것이 신뢰도를 높인다.
- 따라서:
    - 최소 지속시간(min_len) 이상의 연속 예측만 발작으로 인정
    - 발작 사이에 약간의 간격(merge_gap)이 있어도 하나의 episode로 병합
- 이렇게 함으로써 false positive를 줄이고, 진짜 발작 패턴을 안정적으로 포착할 수 있다.


---

✅ **정리**  
- README.md에 요약본 넣고  
- `docs/` 폴더에 위에 정리한 4개 파일 (`preprocessing.md`, `models.md`, `evaluation.md`, `web.md`) 각각 저장하면 완벽해!
