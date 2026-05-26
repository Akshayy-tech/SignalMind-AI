import numpy as np

def generate_signal(signal_type, frequency, amplitude=1.0, sampling_rate=1000, duration=1.0, noise_level=0.1, dc_offset=0.0):
    """
    Generates a clean signal and a noisy signal based on type, frequency, and parameters.
    
    Parameters:
    - signal_type: "Sine Wave" or "Square Wave"
    - frequency: frequency in Hz
    - amplitude: peak amplitude of the clean signal
    - sampling_rate: sample points per second
    - duration: duration in seconds
    - noise_level: standard deviation of Gaussian white noise
    - dc_offset: DC bias added to the signal
    
    Returns:
        t: list of time steps
        clean: list of clean signal values
        noisy: list of noisy signal values
    """
    # Create time vector
    t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)
    
    # Generate clean signal
    if signal_type == "Sine Wave":
        clean = amplitude * np.sin(2 * np.pi * frequency * t) + dc_offset
    elif signal_type == "Square Wave":
        # Scipy square representation can be coded manually to avoid Scipy dependency if we want to be safe,
        # but numpy-based square wave is easy to compute and stable:
        clean = amplitude * np.sign(np.sin(2 * np.pi * frequency * t)) + dc_offset
    else:
        clean = np.zeros_like(t) + dc_offset
        
    # Generate Gaussian white noise
    noise = np.random.normal(0, noise_level, size=t.shape)
    noisy = clean + noise
    
    return t.tolist(), clean.tolist(), noisy.tolist()

def compute_fft(y, sampling_rate):
    """
    Computes Fast Fourier Transform (FFT) of a signal.
    
    Returns:
        frequencies: list of positive frequencies (Hz)
        magnitudes: list of FFT magnitudes scaled to match original amplitude
    """
    y = np.array(y)
    n = len(y)
    
    # Perform FFT
    fft_vals = np.fft.fft(y)
    fft_freq = np.fft.fftfreq(n, 1.0 / sampling_rate)
    
    # Extract the positive frequencies and double the magnitude (single-sided spectrum)
    half_n = n // 2
    freqs = fft_freq[:half_n]
    magnitudes = np.abs(fft_vals[:half_n]) * 2.0 / n
    
    # DC component shouldn't be doubled
    if len(magnitudes) > 0:
        magnitudes[0] = magnitudes[0] / 2.0
        
    return freqs.tolist(), magnitudes.tolist()

def compute_statistics(noisy, clean=None, sampling_rate=1000):
    """
    Calculates Signal-to-Noise Ratio (SNR) and signal health metrics.
    If the clean reference signal is not available (e.g. for uploaded custom signals),
    this function performs spectral estimation of the SNR by analyzing the peak tone
    versus the noise floor in the frequency domain.
    
    Returns:
        dict: SNR (dB), mean, standard deviation, RMS, peak-to-peak, and quality rating (%).
    """
    noisy = np.array(noisy)
    
    if clean is not None:
        clean = np.array(clean)
        noise = noisy - clean
        
        p_signal = np.mean(clean ** 2)
        p_noise = np.mean(noise ** 2)
        
        if p_noise == 0:
            snr = 80.0  # Perfect signal
        elif p_signal == 0:
            snr = -40.0 # Silence/Only noise
        else:
            snr = 10 * np.log10(p_signal / p_noise)
    else:
        # Spectral SNR estimation (No clean reference available)
        n = len(noisy)
        if n > 4:
            fft_vals = np.fft.fft(noisy)
            fft_mag = np.abs(fft_vals[:n//2]) * 2.0 / n
            
            # Find the index of the highest frequency peak (ignoring DC component)
            if len(fft_mag) > 2:
                peak_idx = np.argmax(fft_mag[1:]) + 1
                peak_power = fft_mag[peak_idx] ** 2
                
                # Mask out the signal tone and its immediate spectral neighbors to compute noise floor
                mask = np.ones_like(fft_mag, dtype=bool)
                mask[0] = False  # Exclude DC
                exclude_start = max(1, peak_idx - 2)
                exclude_end = min(len(fft_mag), peak_idx + 3)
                mask[exclude_start:exclude_end] = False
                
                noise_mag = fft_mag[mask]
                if len(noise_mag) > 0:
                    mean_noise_power = np.mean(noise_mag ** 2)
                    if mean_noise_power == 0:
                        snr = 60.0
                    else:
                        snr = 10 * np.log10(peak_power / mean_noise_power)
                else:
                    snr = 20.0
            else:
                snr = 15.0
        else:
            snr = 10.0

    # Calculate time-domain stats
    mean_val = float(np.mean(noisy))
    std_val = float(np.std(noisy))
    rms_val = float(np.sqrt(np.mean(noisy**2)))
    p2p_val = float(np.ptp(noisy))
    
    # Calculate Quality percentage (Quality % based on SNR thresholds)
    # SNR above 25 dB is 100% quality, SNR below -5 dB is 0% quality
    if snr >= 25.0:
        quality = 100.0
    elif snr <= -5.0:
        quality = 0.0
    else:
        quality = (snr - (-5.0)) / (25.0 - (-5.0)) * 100.0
        
    return {
        "snr_db": round(snr, 2),
        "mean": round(mean_val, 4),
        "std_dev": round(std_val, 4),
        "rms": round(rms_val, 4),
        "peak_to_peak": round(p2p_val, 4),
        "quality_pct": round(quality, 1)
    }
