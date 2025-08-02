import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.io.wavfile as wav
from scipy.signal import butter, lfilter
from datetime import datetime
import io
import base64
import tempfile
import google.generativeai as genai

# Configuration
UPLOAD_FOLDER = "uploaded_audios"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Gemini Setup
genai.configure(api_key="AIzaSyDdGv--2i0pMbhH68heurl-LI1qJPJjzD4")
model = genai.GenerativeModel(model_name="gemini-2.5-pro")

# Helpers
def reduce_noise(audio, sr, cutoff):
    nyq = 0.5 * sr
    normal_cutoff = cutoff / nyq
    b, a = butter(2, normal_cutoff, btype='low', analog=False)
    return lfilter(b, a, audio)

def wav_to_bytes(audio_array, sr):
    virtual_file = io.BytesIO()
    wav.write(virtual_file, sr, np.int16(audio_array))
    virtual_file.seek(0)
    return virtual_file.read()

def diagnose_with_gemini_text_from_audio(audio, sr, valve):
    t = np.linspace(0, len(audio) / sr, len(audio))
    fig, ax = plt.subplots()
    ax.plot(t, audio)
    ax.set(title=f"Waveform – {valve}", xlabel="Time (s)", ylabel="Amplitude")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        plt.savefig(tmp.name)
        plt.close(fig)
        image_path = tmp.name

    with open(image_path, "rb") as img_file:
        image_data = base64.b64encode(img_file.read()).decode()

    try:
        gemini_response = model.generate_content([
            {
                "inline_data": {
                    "mime_type": "image/png",
                    "data": image_data
                }
            },
            {
                "text": f"""
You are a medical AI trained to diagnose heart valve conditions from waveform images.
Analyze the waveform image provided for the {valve} valve.
Return a medical report including:
- Diagnosis (e.g., normal, stenosis, regurgitation)
- Severity (mild, moderate, severe)
- Reasoning based on waveform characteristics
- Recommendations
"""
            }
        ])
        return gemini_response.text.strip()
    except Exception as e:
        return f"Gemini diagnosis error: {e}"

def show_waveform(audio, sr, label, color='blue'):
    t = np.linspace(0, len(audio) / sr, len(audio))
    fig, ax = plt.subplots()
    ax.plot(t, audio, color=color)
    ax.set(title=f"{label} Waveform", xlabel="Time (s)", ylabel="Amplitude")
    st.pyplot(fig)

def edit_and_show_waveform(path, label):
    sr, audio = wav.read(path)
    if audio.ndim > 1:
        audio = audio[:, 0]

    st.markdown(f"#### {label} Valve")

    a1, a2, a3 = st.columns(3)
    amp = a1.slider(f"{label} Amplitude", 0.1, 5.0, 1.0, key=f"amp{label}")
    dur = a2.slider(f"{label} Duration (s)", 1, int(len(audio) / sr), 5, key=f"dur{label}")
    cutoff = a3.slider(f"{label} Noise Cutoff", 0.01, 0.5, 0.05, 0.01, key=f"noise{label}")

    adj = audio[:dur * sr] * amp
    filt = reduce_noise(adj, sr, cutoff)

    c1, c2 = st.columns(2)
    with c1:
        st.audio(path)
        show_waveform(audio, sr, f"{label} Original")
    with c2:
        st.audio(io.BytesIO(wav_to_bytes(filt, sr)))
        show_waveform(filt, sr, f"{label} Edited", color='red')

    st.write("##### 🧠 Real-time AI Diagnosis from Waveform")
    ai_img = diagnose_with_gemini_text_from_audio(filt, sr, label)
    st.success(ai_img)

# UI
st.set_page_config(page_title="Real-time Valve Diagnosis", layout="centered")
st.title("🎧 Real-time Heart Valve Diagnosis using pcg ")
st.markdown("Upload `.wav` files and let AI help analyze waveform patterns for valve disorders.")

valve_label = st.selectbox("Select Heart Valve", ["Aortic", "Mitral", "Pulmonary", "Tricuspid"])
uploaded_file = st.file_uploader("Upload .wav file", type=["wav"])

if uploaded_file:
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{valve_label}_{now}.wav"
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    with open(filepath, "wb") as f:
        f.write(uploaded_file.read())

    st.success("File uploaded successfully!")
    edit_and_show_waveform(filepath, valve_label)

# Button Styling
st.markdown("""
    <style>
        .main {
            padding: 20px;
        }
    </style>
""", unsafe_allow_html=True)
