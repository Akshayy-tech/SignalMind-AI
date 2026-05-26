import numpy as np
import re

def analyze_signal_health(y, sampling_rate, clean_reference=None):
    """
    Analyzes the signal to detect noise characteristics using FFT power spectral density.
    
    Returns:
        dict: A health diagnosis including noise type, signal peak frequency, and description.
    """
    y = np.array(y)
    n = len(y)
    
    # Compute FFT to inspect spectral components
    fft_vals = np.fft.fft(y)
    fft_mag = np.abs(fft_vals[:n//2]) * 2.0 / n
    fft_freq = np.fft.fftfreq(n, 1.0 / sampling_rate)[:n//2]
    
    if len(fft_mag) < 4:
        return {
            "noise_type": "Undefined",
            "peak_freq": 0.0,
            "description": "Signal sample size is too small for spectral analysis.",
            "details": "Insufficient data points."
        }
        
    # Find peak frequency (ignoring DC at index 0)
    peak_idx = np.argmax(fft_mag[1:]) + 1
    peak_freq = float(fft_freq[peak_idx])
    peak_amp = float(fft_mag[peak_idx])
    
    # Analyze energy in different bands
    # 1. DC / Low-Freq Drift: frequencies < 15% of peak_freq or < 2 Hz
    low_cutoff = max(2.0, peak_freq * 0.2)
    low_mask = fft_freq < low_cutoff
    low_energy = np.sum(fft_mag[low_mask] ** 2)
    # Exclude DC itself for standard drift, but check it
    dc_value = abs(y.mean())
    
    # 2. Signal Band: around the peak
    sig_band_mask = (fft_freq >= peak_freq * 0.8) & (fft_freq <= peak_freq * 1.2)
    sig_energy = np.sum(fft_mag[sig_band_mask] ** 2)
    
    # 3. High-Freq Noise: frequencies > peak_freq * 1.5 and > 10 Hz
    high_cutoff = max(10.0, peak_freq * 1.5)
    high_mask = fft_freq > high_cutoff
    high_energy = np.sum(fft_mag[high_mask] ** 2)
    
    # Total spectral energy (excluding DC)
    total_energy = np.sum(fft_mag[1:] ** 2)
    
    # Determine noise category
    if total_energy == 0:
        noise_type = "Clean"
        description = "Absolutely clean signal. No noise detected."
        details = "The spectral energy matches only the primary signal components."
    elif clean_reference is not None:
        clean = np.array(clean_reference)
        noise = y - clean
        p_sig = np.mean(clean ** 2)
        p_noise = np.mean(noise ** 2)
        snr = 10 * np.log10(p_sig / p_noise) if p_noise > 0 else 100
        
        if snr > 22.0:
            noise_type = "Clean"
            description = "High quality signal with negligible noise."
            details = f"SNR is very high ({round(snr, 1)} dB). Spectral peaks are sharp and well-defined."
        elif high_energy > sig_energy * 0.15:
            noise_type = "High-Frequency Noise"
            description = "High-frequency ripples or switching noise detected."
            details = "Significant noise energy is present at frequencies above the signal peak. This is typical of thermal noise, electromagnetic interference, or clock ripple."
        elif dc_value > peak_amp * 0.5 or (low_energy > sig_energy * 0.2 and peak_freq > 5):
            noise_type = "Low-Frequency Drift / DC Bias"
            description = "Baseline drift or substantial DC offset detected."
            details = "Low-frequency variations (baseline wander) or a flat DC shift are present. Often caused by temperature fluctuations, sensor drift, or AC coupling issues."
        else:
            noise_type = "Broadband Noise"
            description = "Broadband channel noise / AWGN detected."
            details = "Noise is distributed across the entire spectrum, indicating Additive White Gaussian Noise (AWGN) common in radio channels."
    else:
        # No clean reference, rely purely on spectral ratios
        if high_energy > total_energy * 0.4:
            noise_type = "High-Frequency Noise"
            description = "Predominantly high-frequency noise corrupted."
            details = "FFT shows heavy energy concentrations in high-frequency bins relative to the fundamental tone."
        elif dc_value > peak_amp * 0.6 or (low_energy > total_energy * 0.3 and peak_freq > 8.0):
            noise_type = "Low-Frequency Drift / DC Bias"
            description = "Low-frequency drift or DC bias detected."
            details = "Baseline wander or thermal drift is skewing the signal amplitude center."
        elif total_energy > peak_amp**2 * 1.5:
            noise_type = "Broadband Noise"
            description = "Moderate to heavy broadband background noise."
            details = "The signal peak stands out, but the noise floor is raised across all frequencies."
        else:
            noise_type = "Clean"
            description = "Signal is relatively clean."
            details = "No dominant out-of-band noise or baseline wander was detected."

    return {
        "noise_type": noise_type,
        "peak_freq": round(peak_freq, 2),
        "peak_amp": round(peak_amp, 3),
        "description": description,
        "details": details
    }

def get_filter_recommendation(health_metrics, sampling_rate):
    """
    Formulates a rule-based engineering recommendation for filter design.
    """
    noise_type = health_metrics["noise_type"]
    peak_freq = health_metrics["peak_freq"]
    
    if noise_type == "Clean":
        return {
            "filter_type": "None",
            "cutoff_hz": 0.0,
            "summary": "No filtering is required.",
            "reasoning": "The signal quality is excellent. Applying a filter may unnecessarily attenuate signal components or introduce transit phase shifts."
        }
    elif noise_type == "High-Frequency Noise":
        # Cutoff frequency just above the signal peak
        cutoff = max(1.2 * peak_freq, peak_freq + 2.0)
        cutoff = min(cutoff, 0.45 * sampling_rate) # Keep below Nyquist
        return {
            "filter_type": "Low-Pass Filter",
            "cutoff_hz": round(cutoff, 1),
            "summary": f"Apply a Low-Pass Filter (LPF) with a cutoff of {round(cutoff, 1)} Hz.",
            "reasoning": f"This will pass the fundamental frequency ({peak_freq} Hz) while attenuating the high-frequency noise elements. A Butterworth filter of order 4 is recommended for a flat passband."
        }
    elif noise_type == "Low-Frequency Drift / DC Bias":
        # Cutoff frequency just below the signal peak
        cutoff = min(0.5 * peak_freq, peak_freq - 1.0)
        cutoff = max(cutoff, 0.5) # Don't go to 0
        return {
            "filter_type": "High-Pass Filter",
            "cutoff_hz": round(cutoff, 1),
            "summary": f"Apply a High-Pass Filter (HPF) with a cutoff of {round(cutoff, 1)} Hz.",
            "reasoning": f"This will filter out the DC bias (0 Hz) and slow sensor drifts under {round(cutoff, 1)} Hz, centering the communication signal back around zero."
        }
    else: # Broadband Noise
        # Recommend a Low-Pass Filter slightly above signal frequency to limit noise bandwidth, or a Band-Pass
        cutoff = max(1.25 * peak_freq, peak_freq + 3.0)
        cutoff = min(cutoff, 0.45 * sampling_rate)
        return {
            "filter_type": "Low-Pass Filter",
            "cutoff_hz": round(cutoff, 1),
            "summary": f"Apply a Band-Limiting Low-Pass Filter at {round(cutoff, 1)} Hz.",
            "reasoning": f"For broadband noise, restricting the noise bandwidth (using an LPF cutoff near the signal peak of {peak_freq} Hz) increases the overall SNR by rejecting out-of-band noise energy."
        }

def generate_explanation(signal_type, frequency, snr, quality, health, recommendation):
    """
    Generates a beginner-friendly explanation of the signal diagnostic and filter logic.
    """
    explanation = f"""
### 📊 AI Signal Diagnostic Report

This report explains the characteristics of your generated or uploaded **{signal_type}** signal operating at a nominal frequency of **{frequency} Hz**.

#### 1. Signal Health Metrics
- **Signal-to-Noise Ratio (SNR):** `{snr} dB`
- **Signal Quality:** `{quality}%`
- **Detected Issue:** `{health['noise_type']}` — *{health['description']}*

#### 2. Diagnostic Explanation
{health['details']}

#### 3. AI Engineering Recommendation
- **Recommended Filter:** `{recommendation['filter_type']}`
- **Suggested Cutoff Frequency:** `{recommendation['cutoff_hz']} Hz` (for sampling rate of {frequency * 10 if frequency > 0 else 1000} Hz)
- **Why this filter?** {recommendation['reasoning']}

---
#### 🧠 Communication Systems Insights (For Students)
1. **Nyquist Rate Rule:** Since your signal's peak frequency is `{health['peak_freq']} Hz`, you must sample it at **greater than {round(health['peak_freq'] * 2, 1)} Hz** to avoid *aliasing* (overlapping spectra).
2. **Shannon's Law:** Suppressing noise increases SNR. According to the Shannon-Hartley theorem ($C = B \\log_2(1 + SNR)$), increasing SNR directly expands the maximum error-free data rate (channel capacity) of a communication link. Filtering is essential to achieve modern 5G and Wi-Fi speeds!
"""
    return explanation

def chat_assistant_response(user_query, signal_context=None, openai_api_key=None):
    """
    Handles conversational user queries. If an OpenAI API key is supplied,
    it can query OpenAI. Otherwise, it uses a rich rule-based engine.
    """
    # Clean the query
    q = user_query.strip().lower()
    
    # Format current signal details if available
    context_str = ""
    if signal_context:
        context_str = (
            f"Active Signal: {signal_context.get('type', 'Unknown')} at {signal_context.get('frequency', 0)} Hz. "
            f"SNR: {signal_context.get('snr', 0)} dB. Quality: {signal_context.get('quality', 0)}%. "
            f"Recommended Filter: {signal_context.get('recommended_filter', 'None')}."
        )

    # Optional OpenAI integration if key is provided
    if openai_api_key:
        try:
            import openai
            # We can use the openai SDK if installed. Since we listed it as optional, let's try:
            client = openai.OpenAI(api_key=openai_api_key)
            system_prompt = (
                "You are SignalMind AI, a virtual communications engineering professor and DSP expert. "
                "Help the user understand signal processing, filters (low-pass, high-pass, band-pass), "
                "sampling theorem, FFT, and noise analysis. Keep explanations educational, professional, and friendly. "
                f"Context about current signal: {context_str}"
            )
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_query}
                ],
                max_tokens=250,
                temperature=0.7
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            # Fall back to rule-based system if API call fails
            pass
            
    # --- Robust Rule-Based Expert Chatbot ---
    
    # 1. Check for current signal questions
    if any(k in q for k in ["current signal", "this signal", "my signal", "recommendation", "what filter", "diagnose", "status", "quality"]):
        if signal_context:
            return (
                f"Based on the active signal analysis:\n"
                f"- **Signal Type:** {signal_context.get('type')}\n"
                f"- **Frequency:** {signal_context.get('frequency')} Hz\n"
                f"- **SNR:** {signal_context.get('snr')} dB (Quality: {signal_context.get('quality')}%)\n"
                f"- **Diagnosis:** {signal_context.get('noise_diagnosis')}\n"
                f"- **Recommended Action:** {signal_context.get('recommended_filter')} "
                f"(Cutoff: {signal_context.get('cutoff')} Hz)\n\n"
                f"Applying this filter will attenuate the noise outside the {signal_context.get('frequency')} Hz band, "
                f"which increases your Signal-to-Noise Ratio and improves data demodulation accuracy."
            )
        else:
            return "No signal has been generated or uploaded yet. Please use the sidebar to choose a signal, and I can give you a live analysis!"
            
    # 2. Nyquist / Sampling Theorem
    if any(k in q for k in ["nyquist", "sampling", "alias", "shannon-sampling", "criterion"]):
        return (
            "### The Nyquist-Shannon Sampling Theorem 📈\n\n"
            "This is a fundamental theorem in mixed-signal engineering. It states:\n"
            "> *To perfectly reconstruct a continuous-time signal from its samples, the sampling frequency ($f_s$) "
            "must be strictly greater than twice the highest frequency component ($f_{max}$) present in the signal.*"
            "\n\n"
            "$$f_s > 2 \\cdot f_{max}$$\n\n"
            "If $f_s \\le 2 \\cdot f_{max}$, high-frequency components 'fold back' into the lower spectrum. "
            "This phenomenon is called **Aliasing** and causes irreversible distortion. "
            "To prevent aliasing in real systems, engineers use an analog **Anti-Aliasing Filter** (a low-pass filter) "
            "before the Analog-to-Digital Converter (ADC) to remove components above $f_s/2$ (the Nyquist frequency)."
        )
        
    # 3. FFT / Fourier
    if any(k in q for k in ["fft", "fourier", "dft", "frequency domain", "spectrum", "spectral"]):
        return (
            "### Fast Fourier Transform (FFT) ⚡\n\n"
            "The **Fast Fourier Transform (FFT)** is an efficient algorithm to compute the **Discrete Fourier Transform (DFT)**. "
            "It transforms a signal from the **Time Domain** (amplitude vs. time) to the **Frequency Domain** (amplitude/phase vs. frequency)."
            "\n\n"
            "**Why is it useful?**\n"
            "1. **Isolation:** It's hard to see noise in the time domain, but in the frequency domain, noise is separated by frequency bins.\n"
            "2. **Peak Detection:** We can instantly identify the carrier frequency and harmonic distortions.\n"
            "3. **Filtering Design:** By looking at the FFT, we can see exactly where the noise power lies and design low-pass, high-pass, or band-pass filters to target it."
        )

    # 4. SNR / Decibel
    if any(k in q for k in ["snr", "signal to noise", "decibel", "db", "noise floor"]):
        return (
            "### Signal-to-Noise Ratio (SNR) and Decibels (dB) 🎙️\n\n"
            "**SNR** is a measure used in science and engineering that compares the level of a desired signal to the level of background noise.\n\n"
            "$$SNR = \\frac{P_{signal}}{P_{noise}}$$\n\n"
            "Because signal and noise levels can span multiple orders of magnitude, engineers use the logarithmic **Decibel (dB)** scale:\n"
            "$$SNR_{dB} = 10 \\log_{10}\\left(\\frac{P_{signal}}{P_{noise}}\\right)$$\n\n"
            "**SNR Benchmarks:**\n"
            "- **> 20 dB:** Excellent signal. Crisp communication, high transmission speeds.\n"
            "- **10 to 20 dB:** Moderate signal. Usable, but requires error correction or filtering.\n"
            "- **< 5 dB:** Poor signal. High bit error rate (BER), severe signal corruption.\n"
            "- **Negative dB (e.g. -3 dB):** The noise is stronger than the signal! In GPS or spread spectrum communication, signals can actually be recovered from below the noise floor using coding gains."
        )

    # 5. Low-Pass / High-Pass Filters
    if any(k in q for k in ["low pass", "lpf", "high pass", "hpf", "butterworth", "cutoff"]):
        return (
            "### Filter Fundamentals 🎚️\n\n"
            "Filters alter the amplitude and phase of a signal's frequency components. The two primary types are:\n\n"
            "1. **Low-Pass Filter (LPF):** Allows frequencies below the *cutoff frequency* ($f_c$) to pass while attenuating "
            "frequencies above it. Excellent for eliminating high-frequency fluctuations, clock bleed, and white noise.\n"
            "2. **High-Pass Filter (HPF):** Allows frequencies above the cutoff frequency to pass while blocking DC bias "
            "and low-frequency wander (like breathing drift in ECG signals or thermal fluctuations in radio frontends).\n\n"
            "We use **Butterworth Filters** here, which are known as *maximally flat* magnitude response filters. "
            "This means they have no ripples in the passband, making them perfect for clean signal reconstruction."
        )

    # 6. Shannon Capacity
    if any(k in q for k in ["shannon", "capacity", "theorem", "hartley", "data rate"]):
        return (
            "### Shannon-Hartley Theorem & Channel Capacity 📡\n\n"
            "The **Shannon-Hartley Theorem** defines the absolute maximum rate at which error-free data can be transmitted "
            "over a communication channel of a specified bandwidth ($B$) in the presence of noise:\n\n"
            "$$C = B \\log_2\\left(1 + \\frac{S}{N}\\right)$$\n\n"
            "Where:\n"
            "- $C$ is the **Channel Capacity** in bits per second.\n"
            "- $B$ is the **Bandwidth** in Hertz.\n"
            "- $S/N$ is the linear **Signal-to-Noise Ratio (SNR)**.\n\n"
            "**Engineering Takeaways:**\n"
            "1. To double capacity, you can double the bandwidth $B$ (linear relationship) or significantly increase the signal power $S$ (logarithmic relationship).\n"
            "2. High-speed protocols (like LTE, 5G, Wi-Fi 6) dynamically adjust their modulation (e.g. from QPSK up to 1024-QAM) depending on the real-time SNR. Filtering boosts SNR, which directly speeds up your internet connection!"
        )

    # General greeting / fallback
    return (
        "Hello! I am your **SignalMind AI Assistant** 🎓.\n\n"
        "I can help you understand communication systems and DSP concepts. "
        "Try asking me questions like:\n"
        "- *What is the Nyquist Sampling Theorem?*\n"
        "- *How does the FFT work?*\n"
        "- *What does SNR mean?*\n"
        "- *Explain Low-Pass vs. High-Pass filters.*\n"
        "- *Tell me about Shannon's Channel Capacity.*\n"
        "- *Diagnose my current signal.*"
    )
