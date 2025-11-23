import cv2
import numpy as np
import time
import json
import os
from collections import deque

# Optional: use SciPy if available for better filtering
try:
    from scipy.signal import butter, filtfilt
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False

# ---- Parameters ----
FRAME_WIDTH = 320
FRAME_HEIGHT = 240
HR_FS = 30        # sampling rate for HR signal (samples/sec)
BR_FS = 15        # sampling rate for breathing signal
HR_BAND = (0.7, 3.0)    # 42–180 bpm
BR_BAND = (0.1, 0.5)    # 6–30 brpm
UPDATE_INTERVAL_SEC = 1 # write JSON every 1s
BUFFER_SEC_HR = 20      # window length for HR estimation
BUFFER_SEC_BR = 30      # window length for BR estimation

# ---- Buffers ----
hr_signal = deque(maxlen=int(HR_FS * BUFFER_SEC_HR))
br_signal = deque(maxlen=int(BR_FS * BUFFER_SEC_BR))
hr_values_10s = deque(maxlen=18)  # ~3 minutes of 10s buckets
br_values_10s = deque(maxlen=18)
flow_values_10s = deque(maxlen=18)

start_time = time.time()
last_json_write = 0
last_bucket_time = 0

# ---- Helpers ----
def bandpass_signal(x, fs, low, high):
    if len(x) < fs * 5:
        # Not enough data; just detrend
        x = np.array(x)
        return x - np.mean(x)
    x = np.array(x)
    x = x - np.mean(x)
    if SCIPY_OK:
        b, a = butter(2, [low/(fs/2), high/(fs/2)], btype='band')
        return filtfilt(b, a, x)
    # Fallback: crude moving-average "band" shaping
    # Low-pass by avg window tied to high freq bound; high-pass by detrend
    win = max(1, int(fs/high))
    lp = np.convolve(x, np.ones(win)/win, mode='same')
    return lp

def autocorr_peak_hz(x, fs, min_period=0.3, max_period=3.0):
    x = np.array(x)
    if len(x) < int(fs * min_period) * 3:
        return None
    x = x - np.mean(x)
    # Compute autocorr for lags
    min_lag = int(fs * min_period)
    max_lag = int(fs * max_period)
    best_lag, best_val = -1, -1e9
    for lag in range(min_lag, min(max_lag, len(x)-1)):
        s = np.dot(x[lag:], x[:-lag])
        if s > best_val:
            best_val, best_lag = s, lag
    if best_lag <= 0:
        return None
    period = best_lag / fs
    return 1.0 / period

def gaussian_score(x, mu, sigma):
    if x is None:
        return 0.2
    z = (x - mu) / sigma
    return float(np.exp(-0.5 * z * z))

def stability_score(vals):
    vals = [v for v in vals if v is not None]
    if len(vals) < 3:
        return 0.2
    mean = float(np.mean(vals))
    if mean <= 0:
        return 0.2
    std = float(np.std(vals))
    cv = std / mean
    s = max(0.0, min(1.0, 1.0 - cv))
    return s

def compute_flow(hr_bpm_latest, br_bpm_latest, hr_recent_vals, br_recent_samples):
    # Alignment around typical focused ranges
    hr_align = gaussian_score(hr_bpm_latest, 80.0, 20.0)
    br_align = gaussian_score(br_bpm_latest, 12.0, 6.0)

    # Stability from recent HR tuple values
    hr_stable = stability_score(hr_recent_vals)

    # Regularity from breathing autocorr peak prominence (proxy)
    if len(br_recent_samples) > int(BR_FS * 10):
        s = np.array(br_recent_samples)
        s = s - np.mean(s)
        lag0 = np.dot(s, s)
        max_peak = -1e9
        for lag in range(int(BR_FS * 0.8), int(BR_FS * 6)):
            if lag >= len(s):
                break
            v = np.dot(s[lag:], s[:-lag])
            if v > max_peak:
                max_peak = v
        br_regular = max(0.0, min(1.0, (max_peak / lag0) if lag0 > 0 else 0.0))
    else:
        br_regular = 0.2

    flow01 = 0.35*hr_align + 0.25*hr_stable + 0.25*br_regular + 0.15*br_align
    return int(round(flow01 * 100))

def write_json(bpm, brpm, flow, total_sec, bucket_start):
    payload = {
        "bpm": int(bpm) if bpm is not None else None,
        "breathing_rate": int(brpm) if brpm is not None else None,
        "flow_score": int(flow) if flow is not None else None,
        "t": int(total_sec),
        "bucket": int(bucket_start)  # 10s bucket start
    }
    with open("received_live.json", "w") as f:
        json.dump(payload, f)

# ---- Video capture ----
cap = cv2.VideoCapture(0)  # 0 for default webcam
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
cap.set(cv2.CAP_PROP_FPS, 30)

if not cap.isOpened():
    print("ERROR: Could not open webcam.")
    exit(1)

print("Sensor bridge running. Press 'q' to quit.")

# Timing for sampling
last_hr_sample = 0.0
last_br_sample = 0.0

# For drawing preview windows (optional)
SHOW_WINDOWS = False

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            # If frame read fails, pause briefly
            time.sleep(0.05)
            continue

        # Resize and convert
        frame_small = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
        rgb = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)

        h, w, _ = rgb.shape

        # Define ROIs
        # Forehead: top-center box
        fh_x, fh_y = int(w*0.4), int(h*0.15)
        fh_w, fh_h = int(w*0.2), int(h*0.12)
        forehead = rgb[fh_y:fh_y+fh_h, fh_x:fh_x+fh_w]

        # Chest: mid-lower center box
        ch_x, ch_y = int(w*0.35), int(h*0.55)
        ch_w, ch_h = int(w*0.3), int(h*0.25)
        chest = rgb[ch_y:ch_y+ch_h, ch_x:ch_x+ch_w]

        # Green channel means
        fg = forehead[...,1].astype(np.float32)
        cg = chest[...,1].astype(np.float32)
        forehead_mean = float(np.mean(fg))
        chest_mean = float(np.mean(cg))

        now = time.time()

        # Sample HR at HR_FS
        if now - last_hr_sample >= 1.0 / HR_FS:
            hr_signal.append(forehead_mean)
            last_hr_sample = now

        # Sample BR at BR_FS
        if now - last_br_sample >= 1.0 / BR_FS:
            br_signal.append(chest_mean)
            last_br_sample = now

        # Estimate HR BPM
        hr_bpm = None
        if len(hr_signal) >= int(HR_FS * 8):
            hr_bp = bandpass_signal(list(hr_signal), HR_FS, HR_BAND[0], HR_BAND[1])
            hr_hz = autocorr_peak_hz(hr_bp, HR_FS, 0.3, 3.0)
            if hr_hz:
                hr_bpm = int(round(hr_hz * 60))

        # Estimate Breathing BPM
        br_bpm = None
        if len(br_signal) >= int(BR_FS * 10):
            br_bp = bandpass_signal(list(br_signal), BR_FS, BR_BAND[0], BR_BAND[1])
            br_hz = autocorr_peak_hz(br_bp, BR_FS, 1.0, 6.0)  # broader periods for breathing
            if br_hz:
                br_bpm = int(round(br_hz * 60))

        # 10-second bucketing
        elapsed = int(now - start_time)
        bucket_start = (elapsed // 10) * 10
        if bucket_start != last_bucket_time:
            # Update rolling lists for stability
            if hr_bpm is not None:
                hr_values_10s.append(hr_bpm)
            else:
                hr_values_10s.append(None)

            if br_bpm is not None:
                br_values_10s.append(br_bpm)
            else:
                br_values_10s.append(None)

            # Compute flow
            flow = compute_flow(
                hr_bpm,
                br_bpm,
                [v for v in hr_values_10s if v is not None][-3:],  # last ~30s worth
                list(br_signal)[-int(BR_FS*10):]                    # last 10s raw samples
            )
            flow_values_10s.append(flow)
            last_bucket_time = bucket_start

        # Write JSON every second
        if now - last_json_write >= UPDATE_INTERVAL_SEC:
            flow_latest = flow_values_10s[-1] if len(flow_values_10s) > 0 else None
            write_json(hr_bpm, br_bpm, flow_latest, elapsed, bucket_start)
            last_json_write = now

        # Optional preview windows
        if SHOW_WINDOWS:
            disp = frame_small.copy()
            cv2.rectangle(disp, (fh_x, fh_y), (fh_x+fh_w, fh_y+fh_h), (0,255,0), 1)
            cv2.rectangle(disp, (ch_x, ch_y), (ch_x+ch_w, ch_y+ch_h), (255,0,0), 1)
            txt = f"HR: {hr_bpm if hr_bpm else '--'} | BR: {br_bpm if br_bpm else '--'}"
            cv2.putText(disp, txt, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.imshow("Bio-Canvas Bridge", disp)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

except KeyboardInterrupt:
    pass
finally:
    cap.release()
    if SHOW_WINDOWS:
        cv2.destroyAllWindows()

print("Stopped.")
