Multimodal AI Reading Assistant
Project submitted by: Thanusha Dadi(22118020)
A real-time AI-powered reading support system designed to help learners improve reading accuracy, fluency, and focus.
This prototype integrates speech recognition, computer vision attention tracking, and AI feedback generation into a single interactive Streamlit app.

🚀 Features
🔊 1. Speech Analysis (ASR)
Upload an audio recording of the user reading a sentence.

Automatic speech recognition converts speech to text.

Highlights mispronounced, skipped, or incorrect words.

Computes a reading accuracy score.

👀 2. Real-Time Attention Tracking (Computer Vision)
Using MediaPipe FaceMesh + Iris, the system performs:

Eye blink detection

Gaze direction (left, right, center, up, down)

Basic fatigue estimation

Real-time attention score (0–1 scale)

A continuous live webcam feed displays annotated frames.

🧠 3. Combined Mentor Feedback
Based on:

Reading accuracy (ASR)

Live attention score (CV)

Gaze stability + blink rate

The mentor generates supportive, adaptive feedback for the learner.

🏗️ System Architecture
User Input (Audio + Webcam)
        ↓
SpeechRecognition  ← Audio Processing
        ↓
Text Comparison → Word-Level Mismatch Highlighting
        ↓
Computer Vision Module (MediaPipe FaceMesh + Iris)
        ↓
Attention Score + Gaze + Blink Analysis
        ↓
AI Mentor Feedback (Rule-based)
        ↓
Streamlit UI


▶️ How to Run the App

1. Create virtual environment (recommended)

conda create -n reading python=3.10
conda activate reading

2. Install dependencies

pip install -r requirements.txt
conda install -c conda-forge ffmpeg   # required for audio conversion

3. Run the Streamlit app

streamlit run app.py

The browser will open automatically at localhost:8501.



💻 Technologies Used



Python




Streamlit




OpenCV




MediaPipe (FaceMesh + Iris)




SpeechRecognition




Pydub




Difflib (text comparison)





📘 Purpose of the Project

This project demonstrates how multimodal AI can enhance reading support tools:




Real-time focus monitoring




Pronunciation feedback




Adaptive reading assistance




Logging session metrics




CSV export for parents/teachers




Designed for dyslexic learners, but useful for reading improvement in general.

