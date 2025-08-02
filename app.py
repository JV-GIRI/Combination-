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

# Configuration
UPLOAD_FOLDER = "uploaded_audios"
PATIENT_DATA = "patient_data.json"
SIMULATED_DIAGNOSES = {
    "normal": "**Likely Diagnosis:** Normal Heart Sounds\n\n**Analysis:** Normal S1/S2, no murmurs.",
    "as": "**Likely Diagnosis:** Aortic Stenosis (AS)\n\n**Analysis:** Crescendo‑decrescendo midsystolic murmur.",
    "ar": "**Likely Diagnosis:** Aortic Regurgitation (AR)\n\n**Analysis:** High‑frequency decrescendo diastolic murmur.",
    "ps": "**Likely Diagnosis:** Pulmonary Stenosis (PS)\n\n**Analysis:** Systolic murmur with ejection click.",
    "pr": "**Likely Diagnosis:** Pulmonary Regurgitation (PR)\n\n**Analysis:** Early diastolic murmur with high pitch.",
    "ms": "**Likely Diagnosis:** Mitral Stenosis (MS)\n\n**Analysis:** Low‑frequency diastolic rumbling murmur.",
    "mr": "**Likely Diagnosis:** Mitral Regurgitation (MR)\n\n**Analysis:** Holosystolic blowing murmur.",
    "ts": "**Likely Diagnosis:** Tricuspid Stenosis (TS)\n\n**Analysis:** Diastolic murmur increases with inspiration.",
    "tr": "**Likely Diagnosis:** Tricuspid Regurgitation (TR)\n\n**Analysis:** Holosystolic murmur at lower sternal border."
}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Gemini Setup
genai.configure(api_key="AIzaSyDdGv--2i0pMbhH68heurl-LI1qJPJjzD4")
model = genai.GenerativeModel(model_name="gemini-2.5-flash")

# Data persistence
def save_patient_data(data):
    existing = load_patient_data()
    existing.append(data)
    with open(PATIENT_DATA, "w") as f:
        json.dump(existing, f)

def load_patient_data():
    if os.path.exists(PATIENT_DATA):
        with open(PATIENT_DATA, "r") as f:
            return json.load(f)
    return []

def send_sms(phone_number, message):
    client = Client("AC15ee7441c990e6e8a5afc996ed4a55a1", "6bc0831dae8edb1753ace573a92b6337")
    client.messages.create(body=message, from_="+19096391894", to=phone_number)

# Audio processing helpers
def reduce_noise(audio, sr, cutoff=0.05):
    b, a = butter(6, cutoff)
    return lfilter(b, a, audio)

def wav_to_bytes(audio_data, sample_rate):
    buf = io.BytesIO()
    wav.write(buf, sample_rate, audio_data.astype(np.int16))
    return buf.getvalue()

def show_waveform(audio, sr, label, color='blue'):
    t = np.linspace(0, len(audio)/sr, len(audio))
    fig, ax = plt.subplots()
    ax.plot(t, audio, color=color)
    ax.set(title=f"Waveform – {label}", xlabel="Time (s)", ylabel="Amplitude")
    st.pyplot(fig)

def get_simulated_diagnosis(audio, sr, valve):
    std_dev = np.std(audio); peak_amp = np.max(np.abs(audio))
    z = np.count_nonzero(np.diff(np.sign(audio)))
    if valve=="Aortic":
        if std_dev>4000: return SIMULATED_DIAGNOSES["as"]
        if peak_amp>12000: return SIMULATED_DIAGNOSES["ar"]
    if valve=="Pulmonary":
        if std_dev>3500: return SIMULATED_DIAGNOSES["ps"]
        if peak_amp>11000: return SIMULATED_DIAGNOSES["pr"]
    if valve=="Mitral":
        if 1500<std_dev<=3500: return SIMULATED_DIAGNOSES["ms"]
        if std_dev<=1500 and peak_amp>10000: return SIMULATED_DIAGNOSES["mr"]
    if valve=="Tricuspid":
        if std_dev<1500 and z>7000: return SIMULATED_DIAGNOSES["ts"]
        if peak_amp>8000: return SIMULATED_DIAGNOSES["tr"]
    return SIMULATED_DIAGNOSES["normal"]

def diagnose_with_gemini_text_only(sim_report, valve):
    prompt = f"""
A phonocardiogram waveform for the {valve} valve was analyzed.

{sim_report}

Return only in structured concise format:
- Diagnosis:
- Pathology:
- Waveform:
- Murmur Type:
- Severity:
- Justification:
"""
    try:
        return model.generate_content(prompt).text.strip()
    except Exception as e:
        return f"Error: {e}"

def edit_and_show_waveform(path, label):
    sr, audio = wav.read(path)
    if audio.ndim>1: audio = audio[:,0]
    st.markdown(f"#### {label} Valve")
    a1,a2,a3 = st.columns(3)
    amp = a1.slider(f"{label} Amplitude",0.1,5.0,1.0,key=f"amp{label}")
    dur = a2.slider(f"{label} Duration (s)",1,int(len(audio)/sr),5,key=f"dur{label}")
    cutoff = a3.slider(f"{label} Noise Cutoff",0.01,0.5,0.05,0.01,key=f"noise{label}")
    adj = audio[:dur*sr]*amp
    filt = reduce_noise(adj,sr,cutoff)
    c1, c2 = st.columns(2)
    with c1:
        st.audio(path)
        show_waveform(audio,sr,f"{label} Original")
    with c2:
        st.audio(io.BytesIO(wav_to_bytes(filt,sr)))
        show_waveform(filt,sr,f"{label} Edited",color='red')
    sim = get_simulated_diagnosis(filt,sr,label)
    st.write("##### 🤖 Simulated Report"); st.write(sim)
    ai = diagnose_with_gemini_text_only(sim,label)
    st.write("##### 🧠 AI Diagnosis"); st.success(ai)

# Streamlit UI
st.title("💓 HEARTEST: Giri's AI PCG Analyzer")
st.subheader("🎧 Upload Heart Valve Sounds")
labels=["Aortic","Pulmonary","Tricuspid","Mitral"]
paths={}; cols=st.columns(4)
for i,label in enumerate(labels):
    f = cols[i].file_uploader(f"Upload {label}", type=["wav"], key=f"up{label}")
    if f:
        p = os.path.join(UPLOAD_FOLDER,f"{label}_{f.name}")
        with open(p,"wb") as fp: fp.write(f.getbuffer())
        paths[label]=p

if "patient_saved" not in st.session_state: st.session_state["patient_saved"]=False

with st.sidebar.expander("🗞️ Add Patient Info"):
    name = st.text_input("Name"); age=st.number_input("Age",1,120)
    height=st.number_input("Height (cm)",50.0,250.0); weight=st.number_input("Weight (kg)",2.0,300.0)
    gender=st.radio("Gender",["Male","Female","Other"]); notes=st.text_area("Clinical Notes")
    phone=st.text_input("📾 Patient Phone")

if height and weight:
    bmi = round(weight/((height/100)**2),2); st.markdown(f"**BMI:** {bmi}")

if st.button("📂 Save Patient Case"):
    if len(paths)==4:
        data={"name":name,"age":age,"gender":gender,"notes":notes,"height":height,"weight":weight,"bmi":bmi,
              "files":", ".join(paths.keys()),"date":datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        save_patient_data(data); st.session_state["patient_saved"]=True; st.success("Patient data saved.")
    else:
        st.warning("Upload all 4 valve files.")

if st.session_state["patient_saved"]:
    st.subheader("🔹 Original & Edited Waveforms + AI Diagnosis")
    for label in labels:
        if label in paths:
            edit_and_show_waveform(paths[label],label)

if st.button("📤 Send Case via SMS"):
    if len(paths)==4 and phone:
        msg = f"🌹 PCG Case Summary\nName: {name}\nAge: {age}\nGender: {gender}\nBMI: {bmi}\nNotes: {notes}"
        try: send_sms(phone,msg); st.success("📨 Case sent via SMS.")
        except Exception as e: st.error(f"❌ SMS failed: {e}")
    else:
        st.warning("Complete uploads & phone")

st.subheader("📚 Case History")
hist = load_patient_data()
if hist:
    for entry in reversed(hist):
        with st.expander(f"{entry['name']} ({entry['age']}y) – {entry['date']}"):
            st.write(f"Gender: {entry['gender']} | BMI: {entry.get('bmi','N/A')}")
            st.write(f"Notes: {entry['notes']}")
            for label in labels:
                fn = entry.get("files","").split(", ")[0]
                ap = os.path.join(UPLOAD_FOLDER,f"{label}_{fn}")
                if os.path.exists(ap):
                    st.audio(ap)
                    sr,a = wav.read(ap); a = a[:,0] if a.ndim>1 else a
                    show_waveform(a,sr,f"{label} History")
else:
    st.info("No history records found.")

# Button styling
st.markdown("""
<style>
div.stButton > button:first-child {background-color:#006400; color:white;}
</style>""", unsafe_allow_html=True)
