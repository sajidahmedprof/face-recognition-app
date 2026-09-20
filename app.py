import os
import cv2
import numpy as np
import mediapipe as mp
import streamlit as st
from datetime import datetime

# Page Configuration
st.set_page_config(page_title="Office AI Portal", layout="wide", page_icon="🪪")

FACES_DIR = "faces"
LOG_FILE = "attendance.csv"

if not os.path.exists(FACES_DIR):
    os.makedirs(FACES_DIR)

if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w") as f:
        f.write("Name,Date,Time\n")

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

def log_attendance(name):
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")
    with open(LOG_FILE, "a") as f:
        f.write(f"{name},{date_str},{time_str}\n")

st.title("🛡️ Smart Office Face Detection & Registration System")
tab1, tab2, tab3 = st.tabs(["📸 Self-Registration", "🔍 Gate Scanner", "📊 Logs"])

# TAB 1: REGISTRATION
with tab1:
    st.header("Register New User")
    user_name = st.text_input("Enter Full Name:", placeholder="e.g. Alice Smith")
    img_buffer = st.camera_input("Take Registration Photo")

    if st.button("Save Face Profile", type="primary"):
        if not user_name.strip():
            st.error("Please enter your name.")
        elif img_buffer is None:
            st.error("Please capture a photo first.")
        else:
            bytes_data = img_buffer.getvalue()
            cv_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
            rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            results = face_detection.process(rgb_img)

            if not results.detections:
                st.error("No face detected! Center your face and retake.")
            else:
                clean_name = user_name.strip().replace(" ", "_")
                cv2.imwrite(os.path.join(FACES_DIR, f"{clean_name}.jpg"), cv_img)
                st.success(f"Successfully registered **{user_name}**!")
                st.balloons()

# TAB 2: GATE SCANNER
with tab2:
    st.header("Gate Attendance Verification")
    scan_buffer = st.camera_input("Scan Face for Entry")

    if scan_buffer is not None:
        bytes_data = scan_buffer.getvalue()
        frame = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(rgb_frame)

        if not results.detections:
            st.warning("No face detected.")
        else:
            h, w, _ = frame.shape
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                xmin, ymin = int(bboxC.xmin * w), int(bboxC.ymin * h)
                box_w, box_h = int(bboxC.width * w), int(bboxC.height * h)

                cv2.rectangle(frame, (xmin, ymin), (xmin + box_w, ymin + box_h), (0, 255, 0), 2)
                st.success("✅ Face Detected & Verified!")
            
            st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption="Gate Scanner Output")

# TAB 3: LOGS
with tab3:
    st.header("Attendance Records")
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r") as f:
            st.code(f.read(), language="csv")