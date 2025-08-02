import streamlit as st
import numpy as np
import librosa
import matplotlib.pyplot as plt
import io
import base64
import requests
from scipy.signal import butter, lfilter
from pydub import AudioSegment
from tempfile import NamedTemporaryFile

# ======================== CONFIG ========================
st.set_page_config(page_title="Real-Time PCG AI Diagnosis", layout="wide")
st.title("🔬 Real-Time Heart Sound Diagnosis (PCG + Gemini)")

# ================== HELPER FUNCTIONS ====================
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter(data, lowcut=25.0, highcut=400.0, fs=1000.0, order=4):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def generate_waveform_plot(y, sr):
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(np.linspace(0, len(y)/sr, len(y)), y)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Phonocardiogram Waveform")
    ax.grid(True)
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode()

def convert_audio(file):
    audio = AudioSegment.from_file(file)
    with NamedTemporaryFile(delete=False, suffix=".wav") as f:
        audio.export(f.name, format="wav")
        return f.name

def call_gemini_diagnosis(base64_waveform_img, valve="Mitral"):
    # Sample Gemini prompt
    prompt = {
        "contents": [{
            "role": "user",
            "parts": [
                {
                    "text": f"This is a phonocardiogram waveform image for the {valve} valve. Analyze the waveform and diagnose the likely heart condition. Return only in structured concise format:\n\n- Valve:\n- Condition:\n- Confidence Score (%):\n- Justification:\n- Recommendation:"
                },
                {
                    "inline_data": {
                        "mime_type": "image/png",
                        "data": base64_waveform_img
                    }
                }
            ]
        }]
    }

    # Replace this with actual Gemini API call
    headers = {
        "Authorization": f"Bearer YOUR_GEMINI_API_KEY",
        "Content-Type": "application/json"
    }
    endpoint = "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro-vision:generateContent"
    response = requests.post(endpoint, headers=headers, json=prompt)
    result = response.json()

    if 'candidates' in result:
        return result['candidates'][0]['content']['parts'][0]['text']
    else:
        return "Unable to diagnose. Please try again."

# ================== UPLOAD SECTION =====================
st.subheader("📥 Upload Your Heart Sound (PCG) File")
uploaded_file = st.file_uploader("Upload WAV/MP3/OGG file", type=["wav", "mp3", "ogg"])

if uploaded_file:
    with st.spinner("Processing and diagnosing..."):
        audio_path = convert_audio(uploaded_file)
        y, sr = librosa.load(audio_path, sr=1000)
        y_filtered = bandpass_filter(y, fs=sr)

        # Show waveform
        img_b64 = generate_waveform_plot(y_filtered, sr)
        st.image(f"data:image/png;base64,{img_b64}", use_column_width=True)

        # Diagnosis using Gemini
        diagnosis = call_gemini_diagnosis(img_b64)

        st.markdown("### 📋 Diagnosis Result")
        st.code(diagnosis, language='markdown')

        st.download_button("💾 Download Diagnosis Report", diagnosis, file_name="diagnosis.txt")

# ================== FOOTER STYLING =====================
st.markdown("""
<style>
footer {visibility: hidden;}
section.main > div:has(~ footer) {
    padding-bottom: 0rem;
}
</style>
""", unsafe_allow_html=True)
