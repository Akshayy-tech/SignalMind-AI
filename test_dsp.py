import numpy as np
import signal_processing as sp
import filters as filt
import ai_engine as ai

def test_signal_generation():
    print("Testing signal generation...")
    t, clean, noisy = sp.generate_signal("Sine Wave", frequency=10.0, amplitude=1.0, sampling_rate=1000, duration=1.0, noise_level=0.2)
    assert len(t) == 1000
    assert len(clean) == 1000
    assert len(noisy) == 1000
    print("[OK] Signal generation passed!")
    return t, clean, noisy

def test_fft():
    print("Testing FFT analysis...")
    t, clean, noisy = sp.generate_signal("Sine Wave", frequency=10.0, amplitude=1.0, sampling_rate=1000, duration=1.0, noise_level=0.0)
    freqs, mags = sp.compute_fft(clean, 1000)
    
    # The peak frequency should be close to 10 Hz
    peak_idx = np.argmax(mags[1:]) + 1  # ignore DC
    peak_freq = freqs[peak_idx]
    print(f"Detected peak frequency: {peak_freq} Hz")
    assert abs(peak_freq - 10.0) < 1e-1
    print("[OK] FFT analysis passed!")

def test_filtering():
    print("Testing filter operations...")
    t, clean, noisy = sp.generate_signal("Sine Wave", frequency=5.0, amplitude=1.0, sampling_rate=1000, duration=1.0, noise_level=0.5)
    
    # Apply LPF with cutoff at 15 Hz to remove high-frequency noise
    filtered_lpf = filt.low_pass_filter(noisy, cutoff=15.0, sampling_rate=1000, order=4)
    assert len(filtered_lpf) == 1000
    
    # Calculate SNR before and after LPF
    stats_before = sp.compute_statistics(noisy, clean, 1000)
    stats_after = sp.compute_statistics(filtered_lpf, clean, 1000)
    
    print(f"SNR before filtering: {stats_before['snr_db']} dB")
    print(f"SNR after filtering: {stats_after['snr_db']} dB")
    assert stats_after["snr_db"] > stats_before["snr_db"]
    print("[OK] Filtering operations passed!")

def test_ai_recommendation():
    print("Testing AI engine classification...")
    # Generate signal with high frequency noise
    t = np.linspace(0, 1.0, 1000, endpoint=False)
    clean = np.sin(2 * np.pi * 10 * t)
    # Add a high-freq tone
    noisy = clean + 0.5 * np.sin(2 * np.pi * 200 * t)
    
    diagnosis = ai.analyze_signal_health(noisy, 1000, clean)
    print(f"AI diagnosis: {diagnosis['noise_type']}")
    assert diagnosis["noise_type"] == "High-Frequency Noise"
    
    recommendation = ai.get_filter_recommendation(diagnosis, 1000)
    print(f"AI recommendation: {recommendation['filter_type']} at {recommendation['cutoff_hz']} Hz")
    assert recommendation["filter_type"] == "Low-Pass Filter"
    print("[OK] AI engine passed!")

if __name__ == "__main__":
    print("=== Starting SignalMind AI Unit Tests ===")
    test_signal_generation()
    test_fft()
    test_filtering()
    test_ai_recommendation()
    print("=== All DSP Tests Passed Successfully! ===")
