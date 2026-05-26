import os
import pandas as pd
import numpy as np

def create_sample_signals_directory(directory_path="sample_signals"):
    """
    Creates the sample signals directory and generates a demo telemetry signal CSV.
    This signal simulates a typical telemetry sensor feed with low-frequency drift,
    a 10 Hz sine wave carrier, and a high-frequency interference tone + white noise.
    
    Returns:
        str: path to the created CSV file.
    """
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        
    filepath = os.path.join(directory_path, "sample_telemetry.csv")
    
    # Generate 1 second of signal at 1000 Hz sampling rate (1000 points)
    t = np.linspace(0, 1.0, 1000, endpoint=False)
    
    # 1. Base Carrier Signal: 10 Hz sine wave
    clean = 1.2 * np.sin(2 * np.pi * 10 * t)
    
    # 2. High-Frequency Switching Noise: 150 Hz interference (from power lines or clocks)
    interference = 0.5 * np.sin(2 * np.pi * 150 * t)
    
    # 3. Slow thermal drift / DC wander: 0.8 Hz slow cycle
    drift = 0.3 * np.sin(2 * np.pi * 0.8 * t)
    
    # 4. White noise (thermal noise)
    white_noise = np.random.normal(0, 0.25, size=t.shape)
    
    # Combine to form the noisy telemetry signal
    noisy = clean + interference + drift + white_noise
    
    # Save to a DataFrame and export
    df = pd.DataFrame({
        "Time": t,
        "CleanSignal": clean,
        "NoisySignal": noisy
    })
    
    df.to_csv(filepath, index=False)
    return filepath

def generate_report_markdown(signal_type, frequency, amplitude, noise_level, metrics, diagnosis, recommendation):
    """
    Generates a formal engineering mini-project report.
    This can be rendered in Streamlit or downloaded directly as a Markdown document.
    """
    report = f"""# SignalMind AI - Engineering Analysis Report
**Project Title:** AI-Powered Communication Signal Learning & Analysis Platform  
**Course Code:** EC-ACT Engineering Mini-Project Submission  
**Date:** 2026-05-22  
**Status:** Verification & Testing Completed  

---

## 1. System Parameters
- **Signal Configuration Type:** {signal_type}
- **Nominal Transmitted Frequency:** {frequency} Hz
- **Transmitter Signal Amplitude:** {amplitude} V
- **Configured Channel Noise (Std Dev):** {noise_level}

## 2. Statistical Signal Diagnostics (DSP Metrics)
The signal was analyzed in both time and frequency domains. The results of the statistical calculations are as follows:

| Metric / Parameter | Value | Physical Significance |
| :--- | :--- | :--- |
| **Signal-to-Noise Ratio (SNR)** | **{metrics['snr_db']} dB** | Decibel level of signal power relative to noise power |
| **Signal Quality Rating** | **{metrics['quality_pct']}%** | Normalization indicator for receiver viability |
| **Root Mean Square (RMS)** | **{metrics['rms']} V** | Effective voltage of the composite signal |
| **Peak-to-Peak Amplitude** | **{metrics['peak_to_peak']} V** | Total voltage span (Vmax - Vmin) |
| **Standard Deviation (σ)** | **{metrics['std_dev']} V** | Statistical variance indicating noise intensity |
| **Mean Value (DC Offset)** | **{metrics['mean']} V** | Average voltage level, representing carrier DC offset |

## 3. Intelligent Spectral Diagnosis
- **AI Classification:** `{diagnosis['noise_type']}`
- **Diagnostic Description:** {diagnosis['description']}
- **Spectral Energy Profile:** {diagnosis['details']}
- **Identified Fundamental Peak:** `{diagnosis['peak_freq']} Hz` (at `{diagnosis['peak_amp']} V` spectral amplitude)

## 4. Receiver Filter Recommendations
To reconstruct the signal at the receiver end, the AI Recommendation Engine designed the following digital filter parameters:
- **Recommended Action:** `{recommendation['filter_type']}`
- **Optimal Cutoff Frequency:** `{recommendation['cutoff_hz']} Hz`
- **Filter Order:** `4th Order Butterworth (IIR)`
- **Physical Rationale:** {recommendation['reasoning']}

## 5. Educational and Engineering Conclusions
1. The **{signal_type}** signal was successfully synthesized and subjected to simulated transmission channel noise.
2. Fast Fourier Transform (FFT) analysis successfully mapped the signal to the frequency domain, isolating the fundamental carrier peak from out-of-band noise.
3. Applying the `{recommendation['filter_type']}` filter effectively removes unwanted spectral elements, restoring the signal and boosting the SNR.
4. This experiment validates the critical role of digital signal processing in hardware communication modems to maximize Shannon channel capacity limits.

---
*Report generated automatically by SignalMind AI Expert Engine.*
"""
    return report
