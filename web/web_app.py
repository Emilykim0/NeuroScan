import streamlit as st
from edf_to_npy import edf_to_npy
from sliding_predict import predict_seizure_from_segments
from seizure_utils import analyze_episodes, plot_seizure_interactive
import tempfile
import numpy as np
import os
import base64

# 페이지 설정
st.set_page_config(page_title="NeuroScan", layout="wide", page_icon="🧠")

# 배경 + 스타일 설정
def set_background(image_path):
    with open(image_path, "rb") as f:
        data = f.read()
    b64 = base64.b64encode(data).decode()
    css = f"""
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{b64}");
        background-size: cover;
        background-attachment: fixed;
        background-position: center;
        padding: 2rem;
        color: white;
    }}
    .neuro-header h1 {{
        color: white;
        text-align: center;
        font-size: 3rem;
        margin-bottom: 2rem;
    }}
    .diagnosis-box {{
        background-color: #002B5B;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
    }}
    .status-box {{
        background-color: #004085;
        color: white;
        padding: 1rem;
        border-radius: 10px;
        font-weight: bold;
        margin: 1rem 0;
    }}
    .summary-box {{
        background-color: #002B5B;
        padding: 2rem;
        border-radius: 10px;
        margin: 2rem 0 1rem 0;
        width: 100%;          /* ✅ 전체 폭 */
        max-width: 100%;       /* ✅ 제한 없음 */
        color: white;
        text-align: left;
    }}
    .summary-box h3 {{
        margin-bottom: 1rem;
        font-size: 1.8rem;
    }}
    .summary-box p {{
        margin: 0.5rem 0;
        font-size: 1.1rem;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

set_background("neuron.png")

# ----------------------
# 상태 초기화
# ----------------------
if "uploaded" not in st.session_state:
    st.session_state["uploaded"] = False

# ----------------------
# 상단 제목
# ----------------------
st.markdown('<div class="neuro-header"><h1>🧠 NeuroScan: Seizure Analyzer</h1></div>', unsafe_allow_html=True)

# ----------------------
# 진단 기준 안내
# ----------------------
st.markdown("""
<div class="diagnosis-box">
    <h3>📝 진단 기준</h3>
    <p>🧠 예측된 <strong>뇌전증(발작)</strong> 구간이 <strong>10초 이상</strong> 지속될 경우 발작 에피소드로 간주됩니다.</p>
    <p>🔴 붉은 영역은 발작이 감지된 시점을 나타냅니다.</p>
</div>
""", unsafe_allow_html=True)

# ----------------------
# 파일 업로드 안내문 + 업로더
# ----------------------
st.markdown(
    "<p style='color:white; font-size:1.1rem; text-align:left;'>📁 .edf 파일을 업로드 해주세요</p>",
    unsafe_allow_html=True
)
uploaded_file = st.file_uploader("Upload", type=["edf"])
if uploaded_file:
    st.session_state["uploaded"] = True

# ----------------------
# 분석 진행
# ----------------------
if st.session_state["uploaded"]:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".edf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        edf_path = tmp_file.name

    status = st.empty()
    status.markdown('<div class="status-box">📤 파일 업로드 완료. 분석 중입니다...</div>', unsafe_allow_html=True)

    # EEG 채널 설정
    CHANNELS = [
        "FP1-F7", "F7-T7", "T7-P7", "P7-O1",
        "FP2-F8", "F8-T8", "T8-P8", "P8-O2",
        "FZ-CZ", "CZ-PZ", "FP1-F3", "FP2-F4"
    ]

    # 전처리 및 예측
    segments, raw_signal = edf_to_npy(edf_path, CHANNELS)
    preds, seizure_probs = predict_seizure_from_segments(segments)

    # 슬라이딩 기준 보정
    example_channel = 0
    eeg_continuous = raw_signal[example_channel]
    stride_sec = 2.0
    target_fs = 128
    max_pred_len = len(eeg_continuous) // int(stride_sec * target_fs)
    preds = preds[:max_pred_len]

    # 분석 결과 처리
    seizure_info = analyze_episodes(preds, stride=stride_sec, min_len=5, merge_gap=2.0)
    total_seizure_duration = sum(end - start for start, end in seizure_info["intervals"])

    status.markdown('<div class="status-box">✅ 분석 완료! 결과를 확인하세요.</div>', unsafe_allow_html=True)

    # ----------------------
    # 시각화 출력
    # ----------------------
    st.markdown(f"<h3 style='color:white;'>📊 Seizure Wave (Channel {example_channel})</h3>", unsafe_allow_html=True)
    fig = plot_seizure_interactive(
        eeg_continuous, preds,
        stride=stride_sec,
        fs=target_fs,
        channel_idx=example_channel,
        min_len=5,
        merge_gap=2.0
    )
    st.plotly_chart(fig, use_container_width=True)

    # ----------------------
    # 요약 정보 (summary-box 안에)
    # ----------------------
    st.markdown(f"""
    <div class="summary-box">
        <h3>🔍 예측 결과 요약</h3>
        <p>📌 총 발작 횟수: {seizure_info['count']} 회</p>
        <p>⏱️ 총 발작 시간: {total_seizure_duration:.1f} 초</p>
    </div>
    """, unsafe_allow_html=True)

    # ----------------------
    # 발작 구간 리스트 (summary-box 밖, expander로 따로)
    # ----------------------
    with st.expander("🧾 발작 구간 전체 보기"):
        for idx, (start, end) in enumerate(seizure_info["intervals"]):
            st.markdown(f"<p style='color:white;'>- {idx+1}. {start:.1f}초 ~ {end:.1f}초</p>", unsafe_allow_html=True)
