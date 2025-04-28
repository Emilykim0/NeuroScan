import numpy as np
import plotly.graph_objects as go

def merge_close_intervals(intervals, gap_threshold=4.0):
    """
    연속된 구간 간 간격이 gap_threshold 이하이면 병합
    - intervals: [(start, end), ...]
    - gap_threshold: 병합 기준 간격 (초)
    """
    if not intervals:
        return []

    merged = [intervals[0]]
    for current in intervals[1:]:
        prev_start, prev_end = merged[-1]
        curr_start, curr_end = current

        if curr_start - prev_end <= gap_threshold:
            merged[-1] = (prev_start, max(prev_end, curr_end))  # 병합
        else:
            merged.append(current)

    return merged


def analyze_episodes(preds, stride=2.0, min_len=5, merge_gap=4.0):
    """
    예측된 발작 여부(preds)에서 에피소드 단위로 병합 후 반환
    - stride: segment 간 시간 간격 (초)
    - min_len: 최소 연속된 1의 개수 (즉 최소 발작 지속 시간)
    - merge_gap: episode 사이 병합 허용 gap (초)
    """
    intervals = []
    is_seizing = False
    start_time = None
    duration = 0

    # Step 1. 초기 raw interval들 추출
    for i, p in enumerate(preds):
        time = i * stride
        if p == 1:
            if not is_seizing:
                is_seizing = True
                start_time = time
                duration = 1
            else:
                duration += 1
        else:
            if is_seizing:
                if duration >= min_len:
                    end_time = time
                    intervals.append((start_time, end_time))
                is_seizing = False
                duration = 0

    if is_seizing and duration >= min_len:
        end_time = (len(preds) - 1) * stride
        intervals.append((start_time, end_time))

    # Step 2. 인접한 구간 병합 (merge_gap 이내면 하나로 묶기)
    if len(intervals) <= 1:
        return {"count": len(intervals), "intervals": intervals}

    merged = [intervals[0]]
    for cur_start, cur_end in intervals[1:]:
        prev_start, prev_end = merged[-1]
        if cur_start - prev_end <= merge_gap:
            # 병합
            merged[-1] = (prev_start, cur_end)
        else:
            merged.append((cur_start, cur_end))

    return {"count": len(merged), "intervals": merged}



def plot_seizure_interactive(eeg_np, preds, stride=2.0, fs=128, channel_idx=0, min_len=6, merge_gap=2.0):
    from seizure_utils import analyze_episodes
    seizure_info = analyze_episodes(preds, stride=stride, min_len=min_len, merge_gap=merge_gap)

    total_len = len(eeg_np)
    time = np.linspace(0, total_len / fs, total_len)  # 가로축 시간 계산

    fig = go.Figure()

    # EEG 시그널
    fig.add_trace(go.Scatter(
        x=time, y=eeg_np,
        mode='lines',
        name=f"EEG Channel {channel_idx}",
        line=dict(color="black", width=1)
    ))

    # 붉은 사각형 (발작 감지)
    for start, end in seizure_info['intervals']:
        fig.add_vrect(
            x0=start, x1=end,
            fillcolor="rgba(255, 0, 0, 0.3)",
            line_width=0,
            layer="below",
        )

    fig.update_layout(
        title=f"Seizure Detection Timeline (Interactive)",
        xaxis_title="Time (s)",
        yaxis_title="EEG Amplitude",
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(color="black"),
        xaxis=dict(gridcolor="lightgray"),
        yaxis=dict(gridcolor="lightgray"),
        legend=dict(bgcolor="rgba(255,255,255,0.8)", bordercolor="gray", borderwidth=1),
        margin=dict(t=50, b=40, l=40, r=10),
        height=500
    )

    return fig
