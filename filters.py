import numpy as np
from scipy import signal

def low_pass_filter(y, cutoff, sampling_rate, order=4):
    """
    Applies a zero-phase Butterworth low-pass filter (LPF) to remove high-frequency noise.
    
    Parameters:
    - y: input signal array (list or np.ndarray)
    - cutoff: cutoff frequency in Hz (frequencies above this are attenuated)
    - sampling_rate: sampling frequency of the signal in Hz
    - order: order of the Butterworth filter (higher order = sharper roll-off)
    
    Returns:
        list: filtered signal values
    """
    y = np.array(y)
    nyquist = 0.5 * sampling_rate
    normal_cutoff = cutoff / nyquist
    
    # Boundary checks to prevent filter design errors
    if normal_cutoff >= 1.0:
        normal_cutoff = 0.99  # Cap at just under Nyquist
    elif normal_cutoff <= 0.0:
        return y.tolist()  # Cutoff too low, return original
        
    # Design Butterworth filter
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    
    # Apply zero-phase forward-backward filter to avoid phase distortion
    filtered_y = signal.filtfilt(b, a, y)
    return filtered_y.tolist()

def high_pass_filter(y, cutoff, sampling_rate, order=4):
    """
    Applies a zero-phase Butterworth high-pass filter (HPF) to remove low-frequency drift or DC bias.
    
    Parameters:
    - y: input signal array (list or np.ndarray)
    - cutoff: cutoff frequency in Hz (frequencies below this are attenuated)
    - sampling_rate: sampling frequency of the signal in Hz
    - order: order of the Butterworth filter
    
    Returns:
        list: filtered signal values
    """
    y = np.array(y)
    nyquist = 0.5 * sampling_rate
    normal_cutoff = cutoff / nyquist
    
    # Boundary checks to prevent filter design errors
    if normal_cutoff >= 1.0:
        return np.zeros_like(y).tolist()  # Cutoff exceeds Nyquist, blocks everything
    elif normal_cutoff <= 0.0:
        return y.tolist()  # Cutoff too low, let everything pass
        
    # Design Butterworth filter
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    
    # Apply filter
    filtered_y = signal.filtfilt(b, a, y)
    return filtered_y.tolist()
