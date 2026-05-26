from flask import Flask, request, jsonify
import signal_processing as sp
import filters as filt
import ai_engine as ai

app = Flask(__name__)

@app.route('/api/health', methods=['GET'])
def health():
    """Simple API healthcheck."""
    return jsonify({
        "status": "healthy",
        "service": "SignalMind AI API Service",
        "version": "1.0.0"
    })

@app.route('/api/generate', methods=['POST'])
def generate():
    """Generates signals (sine or square) with noise based on user settings."""
    try:
        data = request.get_json() or {}
        sig_type = data.get("signal_type", "Sine Wave")
        freq = float(data.get("frequency", 10.0))
        amp = float(data.get("amplitude", 1.0))
        fs = int(data.get("sampling_rate", 1000))
        dur = float(data.get("duration", 1.0))
        noise = float(data.get("noise_level", 0.1))
        dc = float(data.get("dc_offset", 0.0))
        
        t, clean, noisy = sp.generate_signal(sig_type, freq, amp, fs, dur, noise, dc)
        return jsonify({
            "success": True,
            "time": t,
            "clean": clean,
            "noisy": noisy
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400

@app.route('/api/process', methods=['POST'])
def process():
    """Computes FFT and standard signal statistics."""
    try:
        data = request.get_json() or {}
        noisy = data.get("noisy", [])
        clean = data.get("clean")  # optional
        fs = int(data.get("sampling_rate", 1000))
        
        if not noisy:
            return jsonify({"success": False, "error": "Noisy signal data is required."}), 400
            
        freqs, mags = sp.compute_fft(noisy, fs)
        metrics = sp.compute_statistics(noisy, clean, fs)
        
        return jsonify({
            "success": True,
            "frequencies": freqs,
            "magnitudes": mags,
            "metrics": metrics
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400

@app.route('/api/filter', methods=['POST'])
def apply_filter():
    """Applies Low-Pass or High-Pass filtering to the provided signal."""
    try:
        data = request.get_json() or {}
        noisy = data.get("noisy", [])
        filter_type = data.get("filter_type", "Low-Pass Filter")
        cutoff = float(data.get("cutoff_hz", 15.0))
        fs = int(data.get("sampling_rate", 1000))
        
        if not noisy:
            return jsonify({"success": False, "error": "Noisy signal data is required."}), 400
            
        if filter_type == "Low-Pass Filter":
            filtered = filt.low_pass_filter(noisy, cutoff, fs)
        elif filter_type == "High-Pass Filter":
            filtered = filt.high_pass_filter(noisy, cutoff, fs)
        else:
            filtered = noisy
            
        return jsonify({
            "success": True,
            "filtered": filtered
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400

@app.route('/api/analyze', methods=['POST'])
def analyze():
    """Runs the expert diagnostics system, computes SNR, and returns explanations."""
    try:
        data = request.get_json() or {}
        noisy = data.get("noisy", [])
        clean = data.get("clean")  # optional
        fs = int(data.get("sampling_rate", 1000))
        sig_type = data.get("signal_type", "Custom Signal")
        freq = float(data.get("frequency", 10.0))
        
        if not noisy:
            return jsonify({"success": False, "error": "Noisy signal data is required."}), 400
            
        diagnosis = ai.analyze_signal_health(noisy, fs, clean)
        recommendation = ai.get_filter_recommendation(diagnosis, fs)
        metrics = sp.compute_statistics(noisy, clean, fs)
        explanation = ai.generate_explanation(sig_type, freq, metrics["snr_db"], metrics["quality_pct"], diagnosis, recommendation)
        
        return jsonify({
            "success": True,
            "diagnosis": diagnosis,
            "recommendation": recommendation,
            "metrics": metrics,
            "explanation": explanation
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400

@app.route('/api/chat', methods=['POST'])
def chat():
    """Handles chatbot integration queries."""
    try:
        data = request.get_json() or {}
        query = data.get("query", "")
        context = data.get("signal_context")
        api_key = data.get("openai_api_key")
        
        reply = ai.chat_assistant_response(query, context, api_key)
        return jsonify({
            "success": True,
            "response": reply
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == '__main__':
    print("Starting Flask SignalMind API Server on http://127.0.0.1:5000")
    app.run(host='127.0.0.1', port=5000, debug=False)
