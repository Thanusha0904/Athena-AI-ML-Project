import cv2
from cv_attention import AttentionDetector
import streamlit as st
import speech_recognition as sr
from pydub import AudioSegment
import difflib
import tempfile
import io
import time
import pandas as pd
from io import BytesIO
# ---------------------------
# Initialize detector + webcam
# ---------------------------
detector = AttentionDetector()
cap = cv2.VideoCapture(0)

st.set_page_config(layout="centered")
st.title("SMART READING ASSISTANT FOR DYSLEXIA")

# ---------------------------
# Sentence Selection
# ---------------------------
SENTENCES = [
    "The sun sets over the quiet lake.",
    "My cat slept on the warm mat."
    "He ran fast to catch the bus",
    "The book is on the table",
    "The children chased the shiny blue ball.",
    "She whispered softly into the night air.",
    "“The weather changes quickly in the mountains.",
    "He thought about the story while walking home.",
    "The three thin thieves thought about the thick thorn.",
    "Shiny shells shimmered softly along the silent seashore.",
    "He rarely realizes how rapidly reality rearranges itself.",
    "The photographer captured the perfect moment under the fading sunlight.",
    "She sells seashells by the seashore.",
    "He threw three stones through the window."
]
sentence = st.selectbox("Choose a sentence", SENTENCES)
st.markdown(f"### Read aloud: **{sentence}**")

# ---------------------------
# REAL COMPUTER VISION MODULE
# ---------------------------
st.subheader("📷 Live Attention Monitoring")

# Create session state flags if not present
if 'cam_running' not in st.session_state:
    st.session_state.cam_running = False

col1, col2 = st.columns([3,1])
with col2:
    if not st.session_state.cam_running:
        if st.button("Start Camera"):
            st.session_state.cam_running = True
    else:
        if st.button("Stop Camera"):
            st.session_state.cam_running = False

if 'attention_history' not in st.session_state:
    # each entry: dict with keys: ts, score, gaze_h, gaze_v, gaze_cat, blink_count
    st.session_state.attention_history = []

# placeholder for the image + small info area
img_placeholder = st.empty()
info_placeholder = st.empty()

# live loop (runs while cam_running is True)
# Note: this is a blocking loop while running, which is fine for a local prototype.
while st.session_state.cam_running:
    ret, frame = cap.read()
    if not ret:
        info_placeholder.write("No camera frame available.")
        break

    info = detector.process_frame(frame)
    annotated = info['annotated_frame']

    # convert to JPEG and show (use width instead of use_column_width)
    _, buf = cv2.imencode('.jpg', annotated)
    img_bytes = buf.tobytes()
    img_placeholder.image(img_bytes, channels='BGR', width=700)
    
    ts = time.time()
    gaze = info.get('gaze') or {}
    sample = {
        'ts': ts,
        'score': float(info.get('attention_score', 0.0)),
        'gaze_h': gaze.get('h') if gaze else None,
        'gaze_v': gaze.get('v') if gaze else None,
        'gaze_h_cat': gaze.get('h_cat') if gaze else None,
        'gaze_v_cat': gaze.get('v_cat') if gaze else None,
        'blink_count': int(info.get('blink_count') or 0),
        'ear': float(info.get('ear') or 0.0)
    }
    st.session_state.attention_history.append(sample)

    # Friendly live info text
    score = sample['score']
    gaze_cat = f"{sample['gaze_h_cat']},{sample['gaze_v_cat']}" if sample['gaze_h_cat'] else "unknown"
    blink = sample['blink_count']

    # recommendation logic
    if score >= 0.75:
        status_text = "Focused — keep going ✅"
        status_color = "#2ecc71"
    elif score >= 0.45:
        status_text = "Partially focused — consider a short break soon ⚠️"
        status_color = "#f1c232"
    else:
        status_text = "Distracted — suggest a short break ⏸️"
        status_color = "#e74c3c"

    info_markdown = (
        f"**Live Attention:** <span style='color:{status_color};font-weight:bold'>{status_text}</span><br><br>"
        f"• **Score:** {score:.2f}<br>"
        f"• **Gaze:** {gaze_cat}<br>"
        f"• **Blink count:** {blink}<br>"
        f"• **Last updated:** {time.strftime('%H:%M:%S', time.localtime(ts))}"
    )
    info_placeholder.markdown(info_markdown, unsafe_allow_html=True)
    # ⬆️⬆️ END OF YOUR BLOCK ⬆️⬆️

    time.sleep(0.05)
    
if not st.session_state.cam_running:
    history = st.session_state.attention_history
    if history:
        df = pd.DataFrame(history)
        df['time_str'] = df['ts'].apply(lambda x: time.strftime('%H:%M:%S', time.localtime(x)))

        st.subheader("📊 Attention Session Summary")

        total_samples = len(df)
        total_time = total_samples * 0.05
        avg_score = df['score'].mean()
        focused_pct = (df[df['score'] >= 0.6].shape[0] / total_samples) * 100

        st.markdown(f"**Duration:** {total_time:.1f} seconds")
        st.markdown(f"**Average score:** {avg_score:.2f}")
        st.markdown(f"**Focused percentage:** {focused_pct:.1f}%")

        st.dataframe(df.tail(20))

        csv_buf = BytesIO()
        df.to_csv(csv_buf, index=False)
        csv_buf.seek(0)
        st.download_button("Download attention session CSV",
                           csv_buf,
                           file_name="attention_session.csv",
                           mime="text/csv")


# when camera is stopped (either by pressing Stop or loop exit), show last frame/info
if not st.session_state.cam_running:
    try:
        # show final frame if exists
        ret, frame = cap.read()
        if ret:
            info = detector.process_frame(frame)
            _, buf = cv2.imencode('.jpg', info['annotated_frame'])
            img_placeholder.image(buf.tobytes(), channels='BGR', width=700)
            info_placeholder.markdown(
                f"**Attention score:** {info['attention_score']}  \n"
                f"**Gaze:** {info.get('gaze')}  \n"
                f"**Blink count (30s window):** {info.get('blink_count')}"
            )
    except Exception:
        pass


# ---------------------------
# Audio Upload + ASR
# ---------------------------
st.subheader("🎤 Upload Your Reading Audio")
st.write("Record the sentence on your phone and upload the audio file (wav).")

uploaded = st.file_uploader("Upload audio", type=["wav","mp3","m4a","ogg"])

if uploaded is not None:
    with st.spinner("Transcribing audio (using Google Web Speech)..."):
        r = sr.Recognizer()
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        data = uploaded.read()

        try:
            audio = AudioSegment.from_file(io.BytesIO(data))
            audio.export(tfile.name, format="wav")
        except Exception:
            tfile.write(data)
            tfile.flush()

        with sr.AudioFile(tfile.name) as source:
            audio_data = r.record(source)
            try:
                transcript = r.recognize_google(audio_data)
            except Exception as e:
                transcript = "[Transcription failed: " + str(e) + "]"

    st.subheader("Transcript (ASR)")
    st.write(transcript)

    # ---------------------------
    # Highlight mismatches
    # ---------------------------
    seq = difflib.SequenceMatcher(None, sentence.lower(), transcript.lower())
    opcodes = seq.get_opcodes()
    highlighted = []
    for tag, i1, i2, j1, j2 in opcodes:
        s = sentence[i1:i2]
        if tag == "equal":
            highlighted.append(s)
        else:
            if s.strip():
                highlighted.append(f"<span style='background:#ffd6d6'>{s}</span>")
            else:
                highlighted.append(s)
    st.markdown("**Expected (mismatches highlighted)**")
    st.markdown(" ".join(highlighted), unsafe_allow_html=True)

    # ---------------------------
    # Mentor Feedback
    # ---------------------------
    score = seq.ratio()
    st.write(f"Match score: {score:.2f} (1.0 = perfect)")

    if score > 0.85:
        mentor = "Amazing! You read most words clearly. Tiny tip: slow down and pronounce 'th' carefully."
    else:
        exp_words = [w.strip(".,").lower() for w in sentence.split()]
        trans_words = [w.strip(".,").lower() for w in transcript.split()]
        misses = [w for w in exp_words if w not in trans_words]
        misses_short = ", ".join(misses[:4]) if misses else "some words"
        mentor = f"Nice try — I noticed trouble with: {misses_short}. Try the 'th' sound: place the tongue between teeth and blow air. Let's try again!"

    st.subheader("Mentor Feedback")
    st.info(mentor)
