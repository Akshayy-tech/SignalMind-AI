# SignalMind AI ⚡
> **AI-Powered Communication Signal Learning & Analysis Platform**  
> *Developed for EC-ACT Engineering Mini-Project Submission*

SignalMind AI is an interactive education and analysis platform that models communication signal generation, transmission noise corruption, spectral analysis, and noise cancellation. The application features a modular full-stack architecture with a **Flask API backend** and a beautiful **Streamlit dashboard frontend**, supported by a **Rule-based AI Expert System** for receiver diagnostics and filter design recommendation.

---

## 🚀 Key Features

1. **Signal Generator (Transmitter Module):**
   - Synthesizes Sine Waves and Square Waves.
   - Configurable Carrier Frequency, Peak Amplitude, DC Offset, Duration, and Sampling Frequency.
   - Models communication channel noise by injecting Gaussian White Noise (AWGN) with adjustable intensity.

2. **Digital Signal Processing (Receiver Module):**
   - **FFT Analysis:** Computes Fast Fourier Transform (FFT) to convert time-domain signals to a single-sided power spectrum.
   - **Statistical Insights:** Calculates RMS voltage, Peak-to-Peak voltage, Standard Deviation (noise intensity), and DC mean.
   - **Advanced SNR Estimation:** Employs a spectral estimation fallback to calculate the Signal-to-Noise Ratio (SNR) in decibels (dB) for uploaded signals when a clean reference is unavailable.

3. **Digital Filtering (Noise Cancellation Module):**
   - Implements zero-phase 1st to 8th order **Butterworth Low-Pass (LPF)** and **High-Pass (HPF)** filters to eliminate spectral noise without phase distortion.

4. **AI Recommendation & Explanation Engine:**
   - Detects the nature of noise: High-Frequency interference, Low-Frequency Drift/DC wander, Broadband Noise, or Clean Signal.
   - Calculates optimal 3dB cutoff frequencies.
   - Explains findings and recommendations in simple, student-friendly English.

5. **Interactive DSP Professor Chatbot:**
   - A rule-based conversational agent teaching core DSP concepts (Nyquist Theorem, FFT, SNR, Shannon Capacity limits, Filters).
   - Context-aware capabilities to analyze the active dashboard signal.

6. **Engineering Report Generator:**
   - Renders a formal laboratory analysis report.
   - Allows users to enter student details (Name, USN, Department, College) to instantly personalize and download the report in Markdown.

---

## 🛠️ Technology Stack
* **Frontend:** Streamlit
* **Backend REST API:** Flask
* **DSP & Analysis:** NumPy, SciPy
* **Data Structures:** Pandas
* **Visualizations:** Plotly (Interactive)

---

## 📁 File Structure
```text
SignalMindAI/
├── app.py                # Streamlit Dashboard Frontend (Auto-spawns server.py)
├── server.py             # Flask REST API Backend
├── signal_processing.py  # Core DSP Calculations (FFT, Statistics, SNR)
├── filters.py            # Butterworth Filter Routines (LPF, HPF)
├── ai_engine.py          # Rule-based AI Diagnostics, Recommendation, & Chatbot
├── utils.py              # Report generation & CSV helpers
├── requirements.txt      # List of dependencies
└── sample_signals/
    └── sample_telemetry.csv  # Auto-generated telemetry sample CSV
```

---

## 🔧 Installation & Setup

1. **Install Dependencies:**
   Ensure you have Python 3.8+ installed, then run:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Application:**
   Start the Streamlit dashboard by executing:
   ```bash
   streamlit run app.py
   ```
   *Note: The Flask Backend API server (`server.py`) will automatically launch in a background thread on `http://127.0.0.1:5000` when the Streamlit application starts.*

---

## 🧠 Educational Insights
- **Nyquist Rate:** $f_s > 2 \cdot f_{max}$. Verify this on the platform by lowering the sampling rate until aliasing occurs in the FFT spectrum!
- **Shannon-Hartley Capacity:** $C = B \log_2(1 + SNR)$. Suppressing noise with LPF increases SNR, boosting communication link capacity.
