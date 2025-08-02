import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.io.wavfile as wav
from scipy.signal import butter, lfilter
from datetime import datetime
import json
import io
from twilio.rest import Client
import google.generativeai as genai
from PIL import Image
import tempfile

# Configuration
UPLOAD_FOLDER = "uploaded_audios"
PATIENT_DATA = "patient_data.json"
SIMULATED_DIAGNOSES = { ... }  # unchanged

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Gemini Setup
genai.configure(api_key="AIzaSyDdGv--2i0pMbhH68heurl-LI1qJPJjzD4")
model = genai.GenerativeModel(model_name="gemini-2.5-flash")

# Data persistence and Twilio helper: unchanged

# Audio processing helpers: unchanged

# Generate waveform image and diagnose with Gemini

def diagnose_with_waveform_image(audio, sr, valve):
    try:
        # Plot waveform and save to temp file
        t = np.linspace(0, len(audio)/sr, len(audio))
        fig, ax = plt.subplots()
        ax.plot(t, audio)
        ax.set(title=f"Waveform – {valve}", xlabel="Time (s)", ylabel="Amplitude")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            plt.savefig(tmp.name)
            plt.close(fig)
            image_path = tmp.name

        # Load image and send to Gemini
        image = Image.open(image_path)
        gemini_response = model.generate_content([
    {
        "inline_data": {
            "mime_type": "image/png",
            "data": base64.b64encode(open(image_path, "rb").read()).decode()
        }
    },
]
        "text": f"""...
                "text": f"""{{ 
  "valve": "{valve}",
  "condition": "Your diagnosis",
  "severity": "Mild/Moderate/Severe",
  "justification": "Brief explanation of waveform features"
}}"""
        ])
        return gemini_response.text.strip()

    except Exception as e:
        return f"Gemini image analysis error: {e}"

# Edit and show waveform + AI integration

def edit_and_show_waveform(path, label):
    sr, audio = wav.read(path)
    if audio.ndim > 1:
        audio = audio[:, 0]
    st.markdown(f"#### {label} Valve")
    a1, a2, a3 = st.columns(3)
    amp = a1.slider(f"{label} Amplitude", 0.1, 5.0, 1.0, key=f"amp{label}")
    dur = a2.slider(f"{label} Duration (s)", 1, int(len(audio)/sr), 5, key=f"dur{label}")
    cutoff = a3.slider(f"{label} Noise Cutoff", 0.01, 0.5, 0.05, 0.01, key=f"noise{label}")
    adj = audio[:dur*sr]*amp
    filt = reduce_noise(adj, sr, cutoff)

    c1, c2 = st.columns(2)
    with c1:
        st.audio(path)
        show_waveform(audio, sr, f"{label} Original")
    with c2:
        st.audio(io.BytesIO(wav_to_bytes(filt, sr)))
        show_waveform(filt, sr, f"{label} Edited", color='red')

    sim = get_simulated_diagnosis(filt, sr, label)
    st.write("##### 🤖 Simulated Report")
    st.write(sim)

    ai_text = diagnose_with_gemini_text_only(sim, label)
    st.write("##### 🧠 AI Diagnosis (Text)")
    st.success(ai_text)

    ai_img = diagnose_with_waveform_image(filt, sr, label)
    st.write("##### 🧠 AI Diagnosis (Waveform Image)")
    st.success(ai_img)

# Streamlit UI, sidebar, saving data, SMS, and history: all unchanged from your code

# Button styling
st.markdown("""
<style>
div.stButton > button:first-child {background-color:#006400; color:white;}
</style>""", unsafe_allow_html=True)
