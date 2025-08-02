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
import json

# Configuration  
UPLOAD_FOLDER = "uploaded_audios"  
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  
CASE_FOLDER = "saved_cases"  
os.makedirs(CASE_FOLDER, exist_ok=True)  
  
# Gemini Setup  
genai.configure(api_key="AIzaSyDdGv--2i0pMbhH68heurl-LI1qJPJjzD4")  
model = genai.GenerativeModel(model_name="gemini-2.5-flash")  
  
# Globals for storing AI summaries per valve  
ai_diagnoses = {}  
  
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
        full_response = gemini_response.text.strip()  
        summary = "\n".join(full_response.split("\n")[:4])  # Brief output (first 3–4 lines)  
        ai_diagnoses[valve] = full_response  # Save full for record  
        return summary  
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
  
    st.markdown(f"### {label} Valve")  
  
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
  
    st.write("#### 🧠 Real-time AI Diagnosis from Waveform")  
    ai_img = diagnose_with_gemini_text_from_audio(filt, sr, label)  
    st.success(ai_img)  
  
# UI  
st.set_page_config(page_title="Real-time Valve Diagnosis", layout="wide")  
st.title("🎷 Real-time Heart Valve Diagnosis using .WAV + AI")  
st.markdown("Upload `.wav` files for each heart valve. The AI will analyze waveform patterns for diagnosis.")  
  
# Sidebar: Patient Info  
st.sidebar.header("🧍‍ Patient Information")  
patient_info = {  
    "Name": st.sidebar.text_input("Name"),  
    "Age": st.sidebar.number_input("Age", min_value=0, max_value=120, step=1),  
    "Gender": st.sidebar.selectbox("Gender", ["Male", "Female", "Other"]),  
    "Height (cm)": st.sidebar.number_input("Height (cm)", min_value=0.0),  
    "Weight (kg)": st.sidebar.number_input("Weight (kg)", min_value=0.0),  
    "Phone": st.sidebar.text_input("Phone Number"),  
    "Clinical Notes": st.sidebar.text_area("Clinical Notes")  
}  
if patient_info["Height (cm)"] > 0 and patient_info["Weight (kg)"] > 0:  
    height_m = patient_info["Height (cm)"] / 100  
    bmi = round(patient_info["Weight (kg)"] / (height_m ** 2), 2)  
    patient_info["BMI"] = bmi  
    st.sidebar.write(f"**BMI:** {bmi}")  
  
# Upload section  
valves = ["Aortic", "Mitral", "Pulmonary", "Tricuspid"]  
uploaded_files = {}  
cols = st.columns(4)  
for i, valve in enumerate(valves):  
    with cols[i]:  
        uploaded_files[valve] = st.file_uploader(f"Upload {valve} Valve .wav", type=["wav"], key=f"upload{valve}")  
  
st.markdown("---")  
  
# Diagnosis section  
for valve in valves:  
    file = uploaded_files[valve]  
    if file:  
        now = datetime.now().strftime("%Y%m%d_%H%M%S")  
        filename = f"{valve}_{now}.wav"  
        filepath = os.path.join(UPLOAD_FOLDER, filename)  
        with open(filepath, "wb") as f:  
            f.write(file.read())  
        edit_and_show_waveform(filepath, valve)  
        st.markdown("---")  
  
# Save Case Button  
if st.button("💾 Save Case"):  
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  
    filename = f"{patient_info['Name'].replace(' ', '_')}_{timestamp}.txt"  
    full_data = {  
        "Patient Info": patient_info,  
        "AI Diagnoses": ai_diagnoses  
    }  
    with open(os.path.join(CASE_FOLDER, filename), "w") as f:  
        f.write(json.dumps(full_data, indent=4))  
    st.success(f"Case saved successfully as `{filename}` in `{CASE_FOLDER}/`")  
  
# Button Styling  
st.markdown("""  
    <style>  
        .main {  
            padding: 20px;  
        }  
    </style>  
""", unsafe_allow_html=True)
