import os
import numpy as np
import pyedflib
from scipy.signal import butter, filtfilt


def bandpass_filter(signal, lowcut=0.5, highcut=40.0, fs=256, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)


def average_reference(signals):
    mean = np.mean(signals, axis=0)
    return signals - mean


def edf_to_npy(edf_path, selected_channels, target_fs=128, epoch_sec=2):
    f = pyedflib.EdfReader(edf_path)
    labels = [ch.strip().upper().replace(" ", "") for ch in f.getSignalLabels()]
    fs = f.getSampleFrequency(0)

    # 채널 필터링 및 인덱스 매핑
    channel_map = {ch: i for i, ch in enumerate(labels)}
    available = [channel_map[ch] for ch in selected_channels if ch in channel_map]
    if len(available) != len(selected_channels):
        raise ValueError("Some selected channels not found in EDF file")

    raw_signals = np.array([f.readSignal(i) for i in available])
    f.close()

    # 평균 참조 및 필터링
    raw_signals = average_reference(raw_signals)
    raw_signals = np.array([bandpass_filter(sig, fs=fs) for sig in raw_signals])

    # 리샘플링
    if fs != target_fs:
        raw_signals = np.array([
            np.interp(
                np.linspace(0, len(sig), int(len(sig) * target_fs / fs)),
                np.arange(len(sig)),
                sig
            ) for sig in raw_signals
        ])

    # 2초 단위 슬라이딩
    window_size = epoch_sec * target_fs
    stride = window_size
    segments = []
    for start in range(0, raw_signals.shape[1] - window_size + 1, stride):
        segment = raw_signals[:, start:start + window_size]
        if segment.shape[1] == window_size:
            segments.append(segment)

    return np.array(segments), raw_signals  # shape: (N, 12, 256), (12, full_len)


if __name__ == "__main__":
    EDF_PATH = "./uploaded_file.edf"
    CHANNELS = [
        "FP1-F7", "F7-T7", "T7-P7", "P7-O1",
        "FP2-F8", "F8-T8", "T8-P8", "P8-O2",
        "FZ-CZ", "CZ-PZ", "FP1-F3", "FP2-F4"
    ]

    segments, raw_signal = edf_to_npy(EDF_PATH, CHANNELS)
    os.makedirs("./UPLOAD", exist_ok=True)
    np.save("./UPLOAD/uploaded_X.npy", segments)
    np.save("./UPLOAD/raw_signal.npy", raw_signal)
    print(f"{segments.shape[0]} segments saved to ./UPLOAD/uploaded_X.npy")
