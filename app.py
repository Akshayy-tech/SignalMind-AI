import streamlit as st
import numpy as np
import pandas as pd
import requests
import json
import time
import socket
import subprocess
import sys
import os
import plotly.graph_objects as go
from utils import create_sample_signals_directory, generate_report_markdown

# --- 1. CONFIGURATION & STYLING ---
st.set_page_config(
    page_title="SignalMind AI - Communication Signal Platform",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Cyberpunk / Modern Dark CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=JetBrains+Mono:wght@400;700&display=swap');
    
    /* Core Typography */
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    code, pre {
        font-family: 'JetBrains Mono', monospace !important;
    }
    
    /* Metric Cards Styling */
    .metric-container {
        display: flex;
        justify-content: space-between;
        gap: 15px;
        margin-bottom: 25px;
        flex-wrap: wrap;
    }
    
    .metric-card {
        flex: 1;
        min-width: 200px;
        background: rgba(17, 24, 39, 0.7);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(0, 255, 204, 0.2);
        border-radius: 16px;
        padding: 22px;
        text-align: center;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.5);
        transition: all 0.3s ease-in-out;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        border-color: rgba(0, 255, 204, 0.6);
        box-shadow: 0 12px 40px 0 rgba(0, 255, 204, 0.15);
    }
    
    .val-good {
        font-size: 34px;
        font-weight: 800;
        color: #00FFCC;
        text-shadow: 0 0 12px rgba(0, 255, 204, 0.4);
    }
    
    .val-mod {
        font-size: 34px;
        font-weight: 800;
        color: #FFCC00;
        text-shadow: 0 0 12px rgba(255, 204, 0, 0.4);
    }
    
    .val-poor {
        font-size: 34px;
        font-weight: 800;
        color: #FF3366;
        text-shadow: 0 0 12px rgba(255, 51, 102, 0.4);
    }
    
    .metric-lbl {
        font-size: 13px;
        color: #8A99AD;
        text-transform: uppercase;
        letter-spacing: 1.8px;
        margin-top: 8px;
        font-weight: 600;
    }
    
    /* Title Glow */
    .title-glow {
        font-size: 3.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #00FFCC 0%, #0077FF 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0 0 30px rgba(0, 255, 204, 0.1);
        margin-bottom: 5px;
    }
    
    .subtitle-text {
        font-size: 1.15rem;
        color: #8A99AD;
        margin-bottom: 30px;
    }
    
    /* Tabs & Streamlit Overrides */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: rgba(17, 24, 39, 0.4);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 8px 8px 0px 0px;
        color: #8A99AD;
        padding: 10px 20px;
        font-weight: 600;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: rgba(0, 255, 204, 0.15) !important;
        border-color: rgba(0, 255, 204, 0.4) !important;
        color: #00FFCC !important;
    }
</style>
""", unsafe_allow_html=True)


# --- 2. BACKEND SERVER MANAGEMENT ---
API_BASE_URL = "http://127.0.0.1:5000"

def is_backend_running():
    """Checks if Flask API server is listening on port 5000."""
    try:
        response = requests.get(f"{API_BASE_URL}/api/health", timeout=1.0)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False

def start_backend():
    """Launches server.py in the background if not running."""
    if not is_backend_running():
        st.info("⚡ Initializing SignalMind AI API Backend...")
        # Start server.py as a background process using system Python interpreter
        # We redirect stdout/stderr to avoid polluting the Streamlit log
        subprocess.Popen(
            [sys.executable, "server.py"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            close_fds=True
        )
        # Wait up to 5 seconds for backend to boot
        for _ in range(10):
            time.sleep(0.5)
            if is_backend_running():
                st.success("✅ Backend API connected successfully!")
                break
        else:
            st.error("❌ Failed to start the Flask Backend. Running in offline/fallback mode.")

# Run backend startup
start_backend()

# Ensure sample telemetry CSV exists
sample_csv_path = create_sample_signals_directory()


# --- 3. STATE INITIALIZATION ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {"role": "assistant", "content": "Welcome to **SignalMind AI Assistant** 🎓. Feel free to ask any questions regarding signals, Nyquist rates, filtering, or to diagnose your current signal!"}
    ]
if "last_ai_cutoff" not in st.session_state:
    st.session_state.last_ai_cutoff = 15.0
if "last_ai_filter" not in st.session_state:
    st.session_state.last_ai_filter = "Low-Pass Filter"


# --- 4. SIDEBAR INPUT CONTROLS ---
st.sidebar.markdown("## 🛠️ Signal Controller")

source_option = st.sidebar.radio("Signal Input Source", ["Synthesize Waveform", "Upload CSV File"])

# Parameter variables
signal_type_val = "Custom Wave"
frequency_val = 10.0
amplitude_val = 1.0
noise_level_val = 0.3
dc_offset_val = 0.0
sampling_rate_val = 1000
duration_val = 1.0

# Store loaded data
uploaded_time = []
uploaded_clean = None
uploaded_noisy = []

if source_option == "Synthesize Waveform":
    signal_type_val = st.sidebar.selectbox("Waveform Type", ["Sine Wave", "Square Wave"])
    frequency_val = st.sidebar.slider("Signal Frequency (Hz)", 1.0, 150.0, 10.0, 1.0)
    amplitude_val = st.sidebar.slider("Peak Amplitude (V)", 0.1, 5.0, 1.0, 0.1)
    noise_level_val = st.sidebar.slider("Noise Intensity (Std Dev)", 0.0, 2.0, 0.35, 0.05)
    dc_offset_val = st.sidebar.slider("DC Offset (V)", -2.0, 2.0, 0.0, 0.1)
    sampling_rate_val = st.sidebar.slider("Sampling Frequency (Hz)", 200, 2500, 1000, 100)
    duration_val = st.sidebar.slider("Signal Duration (sec)", 0.2, 5.0, 1.0, 0.1)
    
    # Generate generated signal via API
    try:
        payload = {
            "signal_type": signal_type_val,
            "frequency": frequency_val,
            "amplitude": amplitude_val,
            "sampling_rate": sampling_rate_val,
            "duration": duration_val,
            "noise_level": noise_level_val,
            "dc_offset": dc_offset_val
        }
        res = requests.post(f"{API_BASE_URL}/api/generate", json=payload)
        if res.status_code == 200:
            res_data = res.json()
            uploaded_time = res_data["time"]
            uploaded_clean = res_data["clean"]
            uploaded_noisy = res_data["noisy"]
        else:
            st.error("API error during signal generation")
    except Exception as e:
        # Fallback to local numpy implementation if API is down
        t = np.linspace(0, duration_val, int(sampling_rate_val * duration_val), endpoint=False)
        if signal_type_val == "Sine Wave":
            clean = amplitude_val * np.sin(2 * np.pi * frequency_val * t) + dc_offset_val
        else:
            clean = amplitude_val * np.sign(np.sin(2 * np.pi * frequency_val * t)) + dc_offset_val
        noise = np.random.normal(0, noise_level_val, size=t.shape)
        uploaded_time = t.tolist()
        uploaded_clean = clean.tolist()
        uploaded_noisy = (clean + noise).tolist()

else:
    # CSV upload mechanism
    st.sidebar.markdown("### Upload Telemetry Signal")
    uploaded_file = st.sidebar.file_uploader("Choose CSV file (Max 5MB)", type=["csv"])
    
    # Offer template file download
    with open(sample_csv_path, "r") as f:
        csv_data = f.read()
    st.sidebar.download_button(
        label="📥 Download Sample Telemetry CSV",
        data=csv_data,
        file_name="sample_telemetry.csv",
        mime="text/csv"
    )
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.sidebar.success("CSV loaded successfully!")
            cols = df.columns.tolist()
            
            time_col = st.sidebar.selectbox("Select Time Column", cols, index=0 if "Time" in cols else 0)
            noisy_col = st.sidebar.selectbox("Select Signal/Noisy Column", cols, index=2 if "NoisySignal" in cols else (1 if len(cols) > 1 else 0))
            
            # Check if clean exists for reference
            clean_col = None
            if len(cols) > 2:
                has_clean = st.sidebar.checkbox("Has Clean Signal reference?", value=("CleanSignal" in cols or "Signal" in cols))
                if has_clean:
                    clean_col = st.sidebar.selectbox("Select Clean Signal Column", cols, index=1 if "CleanSignal" in cols else 1)
            
            sampling_rate_val = st.sidebar.number_input("Sampling Frequency (Hz)", value=1000, min_value=1)
            
            # Populate data arrays
            uploaded_time = df[time_col].tolist()
            uploaded_noisy = df[noisy_col].tolist()
            if clean_col:
                uploaded_clean = df[clean_col].tolist()
                
            signal_type_val = "Uploaded CSV Signal"
            # Estimate frequency roughly for reporting
            diffs = np.diff(np.array(uploaded_time))
            if len(diffs) > 0:
                sampling_rate_val = int(round(1.0 / np.mean(diffs)))
                
        except Exception as e:
            st.sidebar.error(f"Error parsing file: {e}")
    else:
        # Load default telemetry demo signal
        df = pd.read_csv(sample_csv_path)
        uploaded_time = df["Time"].tolist()
        uploaded_clean = df["CleanSignal"].tolist()
        uploaded_noisy = df["NoisySignal"].tolist()
        signal_type_val = "Simulated Telemetry (Default)"
        sampling_rate_val = 1000

# --- FILTER CONTROLLER ---
st.sidebar.markdown("## 🎚️ Digital Filter Design")
filter_type_val = st.sidebar.selectbox("Active Filter Type", ["None", "Low-Pass Filter", "High-Pass Filter"])
cutoff_val = st.sidebar.slider("Cutoff Frequency (Hz)", 0.5, float(sampling_rate_val // 2) - 1.0, 15.0, 0.5)
filter_order_val = st.sidebar.slider("Butterworth Filter Order", 1, 8, 4)


# --- 5. DATA API PROCESSING ---
# Perform FFT & stats calculation
fft_freqs = []
fft_mags = []
metrics = {"snr_db": 0.0, "mean": 0.0, "std_dev": 0.0, "rms": 0.0, "peak_to_peak": 0.0, "quality_pct": 0.0}
diagnosis = {"noise_type": "Clean", "peak_freq": 0.0, "peak_amp": 0.0, "description": "", "details": ""}
recommendation = {"filter_type": "None", "cutoff_hz": 0.0, "summary": "", "reasoning": ""}
explanation_text = ""

try:
    # Process FFT and Stats
    proc_payload = {
        "noisy": uploaded_noisy,
        "clean": uploaded_clean,
        "sampling_rate": sampling_rate_val
    }
    proc_res = requests.post(f"{API_BASE_URL}/api/process", json=proc_payload)
    if proc_res.status_code == 200:
        p_data = proc_res.json()
        fft_freqs = p_data["frequencies"]
        fft_mags = p_data["magnitudes"]
        metrics = p_data["metrics"]
        
    # Process AI Diagnosis & recommendations
    diag_payload = {
        "noisy": uploaded_noisy,
        "clean": uploaded_clean,
        "sampling_rate": sampling_rate_val,
        "signal_type": signal_type_val,
        "frequency": frequency_val if source_option == "Synthesize Waveform" else float(metrics["mean"]) # approximation
    }
    diag_res = requests.post(f"{API_BASE_URL}/api/analyze", json=diag_payload)
    if diag_res.status_code == 200:
        d_data = diag_res.json()
        diagnosis = d_data["diagnosis"]
        recommendation = d_data["recommendation"]
        explanation_text = d_data["explanation"]
        
        # Store recommendations in session state for instant clicking
        st.session_state.last_ai_cutoff = recommendation["cutoff_hz"]
        st.session_state.last_ai_filter = recommendation["filter_type"]
except Exception as e:
    # API Offline fallback calculations
    import signal_processing as fallback_sp
    import ai_engine as fallback_ai
    
    fft_freqs, fft_mags = fallback_sp.compute_fft(uploaded_noisy, sampling_rate_val)
    metrics = fallback_sp.compute_statistics(uploaded_noisy, uploaded_clean, sampling_rate_val)
    diagnosis = fallback_ai.analyze_signal_health(uploaded_noisy, sampling_rate_val, uploaded_clean)
    recommendation = fallback_ai.get_filter_recommendation(diagnosis, sampling_rate_val)
    explanation_text = fallback_ai.generate_explanation(
        signal_type_val, 
        frequency_val if source_option == "Synthesize Waveform" else diagnosis["peak_freq"],
        metrics["snr_db"], metrics["quality_pct"], diagnosis, recommendation
    )

# Filter the signal
filtered_signal = list(uploaded_noisy)
if filter_type_val != "None":
    try:
        filt_payload = {
            "noisy": uploaded_noisy,
            "filter_type": filter_type_val,
            "cutoff_hz": cutoff_val,
            "sampling_rate": sampling_rate_val
        }
        filt_res = requests.post(f"{API_BASE_URL}/api/filter", json=filt_payload)
        if filt_res.status_code == 200:
            filtered_signal = filt_res.json()["filtered"]
    except Exception as e:
        import filters as fallback_filt
        if filter_type_val == "Low-Pass Filter":
            filtered_signal = fallback_filt.low_pass_filter(uploaded_noisy, cutoff_val, sampling_rate_val, filter_order_val)
        else:
            filtered_signal = fallback_filt.high_pass_filter(uploaded_noisy, cutoff_val, sampling_rate_val, filter_order_val)


# --- 6. FRONTEND PRESENTATION LAYER ---
st.markdown('<div class="title-glow">⚡ SignalMind AI</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle-text">AI-Powered Communication Signal Learning & Analysis Platform (EC Engineering Mini-Project)</div>', unsafe_allow_html=True)

# Metric Summary Ribbon
qual_pct = metrics["quality_pct"]
qual_class = "val-good" if qual_pct >= 75 else ("val-mod" if qual_pct >= 40 else "val-poor")

snr_db = metrics["snr_db"]
snr_class = "val-good" if snr_db >= 20 else ("val-mod" if snr_db >= 8 else "val-poor")

rec_filter = recommendation["filter_type"]
rec_class = "val-good" if rec_filter == "None" else "val-mod"

st.markdown(f"""
<div class="metric-container">
    <div class="metric-card">
        <div class="{qual_class}">{qual_pct}%</div>
        <div class="metric-lbl">Signal Quality</div>
    </div>
    <div class="metric-card">
        <div class="{snr_class}">{snr_db} dB</div>
        <div class="metric-lbl">Signal-to-Noise Ratio</div>
    </div>
    <div class="metric-card">
        <div class="val-good">{diagnosis['peak_freq']} Hz</div>
        <div class="metric-lbl">Detected Peak Freq</div>
    </div>
    <div class="metric-card">
        <div class="{rec_class}">{rec_filter}</div>
        <div class="metric-lbl">AI Recommendation</div>
    </div>
</div>
""", unsafe_allow_html=True)


# TABS STRUCTURE
tab_plots, tab_ai, tab_chat, tab_report = st.tabs([
    "📊 Signals & Spectrum Dashboard", 
    "🧠 AI Diagnostic Insights", 
    "💬 Interactive DSP Professor Chat", 
    "📄 Mini-Project Report Summary"
])

# --- TAB 1: DASHBOARD ---
with tab_plots:
    col1, col2 = st.columns(2)
    
    with col1:
        # Time Domain Plot
        fig_time = go.Figure()
        fig_time.add_trace(go.Scatter(
            x=uploaded_time[:500], y=uploaded_noisy[:500],
            name="Noisy Channel Signal", line=dict(color='#FF3366', width=1.5)
        ))
        if uploaded_clean is not None:
            fig_time.add_trace(go.Scatter(
                x=uploaded_time[:500], y=uploaded_clean[:500],
                name="Ideal Carrier Signal", line=dict(color='#00FFCC', width=2.5)
            ))
        fig_time.update_layout(
            title="Time Domain Waveform (Zoomed: First 500 samples)",
            xaxis_title="Time (seconds)",
            yaxis_title="Amplitude (Volts)",
            template="plotly_dark",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=20, r=20, t=50, b=20)
        )
        st.plotly_chart(fig_time, use_container_width=True)

    with col2:
        # Frequency Domain Plot (FFT)
        fig_freq = go.Figure()
        # Crop frequencies above Nyquist (which is already done in compute_fft)
        fig_freq.add_trace(go.Bar(
            x=fft_freqs, y=fft_mags,
            name="FFT Spectrum", marker_color='#B5179E'
        ))
        
        # Add a threshold line or vertical peak marker
        if diagnosis["peak_freq"] > 0:
            fig_freq.add_vline(
                x=diagnosis["peak_freq"], line_dash="dash", line_color="#00FFCC",
                annotation_text=f"Peak: {diagnosis['peak_freq']} Hz", annotation_position="top right"
            )
            
        fig_freq.update_layout(
            title="FFT Power Spectrum (Frequency Domain)",
            xaxis_title="Frequency (Hz)",
            yaxis_title="Spectral Amplitude",
            template="plotly_dark",
            margin=dict(l=20, r=20, t=50, b=20)
        )
        st.plotly_chart(fig_freq, use_container_width=True)
        
    st.markdown("---")
    
    # Bottom Row: Filter Recovery Results
    fcol1, fcol2 = st.columns([2, 1])
    
    with fcol1:
        # Filtered Waveform comparison
        fig_filt = go.Figure()
        fig_filt.add_trace(go.Scatter(
            x=uploaded_time[:500], y=uploaded_noisy[:500],
            name="Raw Noisy Input", line=dict(color='rgba(255, 51, 102, 0.4)', width=1)
        ))
        fig_filt.add_trace(go.Scatter(
            x=uploaded_time[:500], y=filtered_signal[:500],
            name="Filtered Output", line=dict(color='#00FF87', width=2.5)
        ))
        if uploaded_clean is not None:
            fig_filt.add_trace(go.Scatter(
                x=uploaded_time[:500], y=uploaded_clean[:500],
                name="Original Ideal", line=dict(color='rgba(0, 255, 204, 0.6)', width=1.5, dash='dash')
            ))
            
        status_txt = f"[{filter_type_val} applied at {cutoff_val} Hz]" if filter_type_val != "None" else "[No Active Filter]"
        fig_filt.update_layout(
            title=f"Time Domain Reconstruction {status_txt}",
            xaxis_title="Time (seconds)",
            yaxis_title="Amplitude (Volts)",
            template="plotly_dark",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=20, r=20, t=50, b=20)
        )
        st.plotly_chart(fig_filt, use_container_width=True)
        
    with fcol2:
        st.markdown("### 🎛️ Live Diagnostics")
        st.markdown(f"**Current Noise Characteristic:** `{diagnosis['noise_type']}`")
        st.markdown(f"**Mean / DC Level:** `{metrics['mean']} V`")
        st.markdown(f"**RMS Value:** `{metrics['rms']} V`")
        st.markdown(f"**Signal Variance (Std Dev):** `{metrics['std_dev']} V`")
        
        # Interactive application button
        st.markdown("#### ⚡ AI Quick Configuration")
        if recommendation["filter_type"] != "None":
            st.info(f"AI suggests applying a **{recommendation['filter_type']}** with cutoff at **{recommendation['cutoff_hz']} Hz**.")
            st.markdown("*Use the sidebar controls to set these parameters, or hit the manual override.*")
        else:
            st.success("Signal is clean! No filtering actions required.")

# --- TAB 2: AI INSIGHTS ---
with tab_ai:
    st.markdown(explanation_text)
    
    st.markdown("### 🛠️ Interactive AI Action Panel")
    act_col1, act_col2 = st.columns(2)
    with act_col1:
        st.write("Would you like to instantly align the hardware simulator controls with the AI recommended filter?")
        if st.button("🚀 Align Simulator Filter Settings", help="Synchronize the sidebar filters with AI design parameters"):
            st.markdown(
                f"<div style='color:#00FFCC; font-weight:bold; margin-bottom:10px;'>"
                f"✅ Action Taken: Please verify sidebar selection updates to {st.session_state.last_ai_filter} "
                f"and Cutoff {st.session_state.last_ai_cutoff} Hz!</div>", 
                unsafe_allow_html=True
            )
            st.warning("👈 Note: Please manually select the suggested parameters in the sidebar to visualize the live change!")
            
    with act_col2:
        st.write("Learn how signal-to-noise ratio limits channel throughput:")
        # Display the Shannon limit calculator
        sh_bw = st.slider("Channel Bandwidth (kHz)", 1, 100, 10, 1)
        sh_snr_linear = 10 ** (metrics["snr_db"] / 10.0)
        capacity_kbps = sh_bw * np.log2(1 + sh_snr_linear)
        st.metric(
            label="Shannon Channel Capacity (kbps)",
            value=f"{round(capacity_kbps, 2)} kbps",
            delta=f"Based on {metrics['snr_db']} dB SNR"
        )


# --- TAB 3: CHAT ASSISTANT ---
with tab_chat:
    st.markdown("### 💬 Learn DSP with the AI Professor")
    st.write("Ask questions about the current signal, Nyquist rate, sampling, FFT or filters below.")

    # Display Chat Log
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            
    # User response box
    user_query = st.chat_input("Type your question here (e.g. 'What is Nyquist criteria?', 'Explain my SNR')")
    
    if user_query:
        # Show user message
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)
            
        # Call API for chat reply
        reply_content = ""
        try:
            # Build current signal context block for context-aware answers
            sig_ctx = {
                "type": signal_type_val,
                "frequency": frequency_val if source_option == "Synthesize Waveform" else diagnosis["peak_freq"],
                "snr": metrics["snr_db"],
                "quality": metrics["quality_pct"],
                "noise_diagnosis": diagnosis["noise_type"],
                "recommended_filter": recommendation["filter_type"],
                "cutoff": recommendation["cutoff_hz"]
            }
            
            chat_payload = {
                "query": user_query,
                "signal_context": sig_ctx,
                "openai_api_key": st.session_state.get("openai_key")
            }
            
            # API query
            res = requests.post(f"{API_BASE_URL}/api/chat", json=chat_payload)
            if res.status_code == 200:
                reply_content = res.json()["response"]
            else:
                reply_content = "API Chat Error: Received bad status code from backend server."
        except Exception as e:
            # Fallback local chat logic
            import ai_engine as fallback_ai
            sig_ctx = {
                "type": signal_type_val,
                "frequency": frequency_val if source_option == "Synthesize Waveform" else diagnosis["peak_freq"],
                "snr": metrics["snr_db"],
                "quality": metrics["quality_pct"],
                "noise_diagnosis": diagnosis["noise_type"],
                "recommended_filter": recommendation["filter_type"],
                "cutoff": recommendation["cutoff_hz"]
            }
            reply_content = fallback_ai.chat_assistant_response(user_query, sig_ctx, st.session_state.get("openai_key"))
            
        # Append and display response
        st.session_state.chat_history.append({"role": "assistant", "content": reply_content})
        with st.chat_message("assistant"):
            st.markdown(reply_content)

# --- TAB 4: REPORT SUMMARY ---
with tab_report:
    st.markdown("### 📝 Generate Engineering Mini-Project Report")
    st.write("Complete the details below to generate a pre-formatted laboratory report ready for submission.")
    
    rep_col1, rep_col2 = st.columns(2)
    with rep_col1:
        student_name = st.text_input("Student Full Name", value="Akshay S")
        roll_no = st.text_input("University Roll Number / USN", value="1KT22EC001")
    with rep_col2:
        dept_name = st.text_input("Department Name", value="Electronics & Communication Engineering")
        inst_name = st.text_input("Institution / College Name", value="ACT Engineering College")
        
    # Build report content
    raw_report = generate_report_markdown(
        signal_type_val,
        frequency_val if source_option == "Synthesize Waveform" else diagnosis["peak_freq"],
        amplitude_val if source_option == "Synthesize Waveform" else 1.0,
        noise_level_val if source_option == "Synthesize Waveform" else metrics["std_dev"],
        metrics,
        diagnosis,
        recommendation
    )
    
    # Inject student metadata
    meta_block = f"""# lab-report
**Candidate:** {student_name} ({roll_no})  
**Department:** {dept_name}  
**Institution:** {inst_name}  
"""
    final_report = raw_report.replace("# SignalMind AI - Engineering Analysis Report", f"# SignalMind AI - Engineering Analysis Report\n{meta_block}")
    
    st.markdown("---")
    st.markdown("### Report Preview")
    st.markdown(final_report)
    
    st.download_button(
        label="📥 Download Engineering Lab Report (Markdown)",
        data=final_report,
        file_name=f"SignalMindAI_Report_{roll_no}.md",
        mime="text/markdown"
    )
