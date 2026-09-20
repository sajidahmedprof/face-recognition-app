import os
import cv2
import numpy as np
import mediapipe as mp
import streamlit as st
import face_recognition
from datetime import datetime

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(page_title="Office AI Portal", layout="wide", page_icon="🪪")

FACES_DIR = "faces"
LOG_FILE = "attendance.csv"

# Ensure direct access paths exist
os.makedirs(FACES_DIR, exist_ok=True)

if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w") as f:
        f.write("Name,Date,Time\n")

# ---------------------------------------------------------
# Cached Resource Loaders (Prevents Memory Leak on Re-runs)
# ---------------------------------------------------------
@st.cache_resource
def load_face_detector():
    """Load MediaPipe Face Detection model into memory once."""
    mp_face = mp.solutions.face_detection
    return mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.5)

face_detection = load_face_detector()

def load_known_faces():
    """Load and compute embeddings for all registered faces in FACES_DIR."""
    known_encodings = []
    known_names = []
    
    if not os.path.exists(FACES_DIR):
        return known_encodings, known_names

    for filename in os.listdir(FACES_DIR):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            filepath = os.path.join(FACES_DIR, filename)
            image = face_recognition.load_image_file(filepath)
            encodings = face_recognition.face_encodings(image)
            
            if encodings:
                known_encodings.append(encodings[0])
                # Clean filename to extract user name (e.g., 'Alice_Smith.jpg' -> 'Alice Smith')
                name = os.path.splitext(filename)[0].replace("_", " ")
                known_names.append(name)

    return known_encodings, known_names

def log_attendance(name):
    """Log user verification to CSV file."""
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")
    with open(LOG_FILE, "a") as f:
        f.write(f"{name},{date_str},{time_str}\n")

# ---------------------------------------------------------
# Application UI & Logic
# ---------------------------------------------------------
st.title("🛡️ Smart Office Face Detection & Registration System")
tab1, tab2, tab3 = st.tabs(["📸 Self-Registration", "🔍 Gate Scanner", "📊 Logs"])

# =========================================================
# TAB 1: REGISTRATION
# =========================================================
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
            
            # MediaPipe Detection Step
            results = face_detection.process(rgb_img)

            if not results.detections:
                st.error("No face detected! Center your face and retake.")
            else:
                # Calculate 128-d encoding to verify face readability
                encodings = face_recognition.face_encodings(rgb_img)
                if not encodings:
                    st.error("Face detected, but features could not be extracted cleanly. Please ensure good lighting.")
                else:
                    clean_name = user_name.strip().replace(" ", "_")
                    file_path = os.path.join(FACES_DIR, f"{clean_name}.jpg")
                    cv2.imwrite(file_path, cv_img)
                    st.success(f"Successfully registered **{user_name}**!")
                    st.balloons()

# =========================================================
# TAB 2: GATE SCANNER
# =========================================================
with tab2:
    st.header("Gate Attendance Verification")
    scan_buffer = st.camera_input("Scan Face for Entry")

    if scan_buffer is not None:
        bytes_data = scan_buffer.getvalue()
        frame = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect bounding boxes via MediaPipe
        results = face_detection.process(rgb_frame)

        if not results.detections:
            st.warning("No face detected.")
        else:
            # Load stored face encodings
            known_encodings, known_names = load_known_faces()
            
            # Extract encoding for the current scan frame
            scan_encodings = face_recognition.face_encodings(rgb_frame)

            h, w, _ = frame.shape
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                xmin, ymin = int(bboxC.xmin * w), int(bboxC.ymin * h)
                box_w, box_h = int(bboxC.width * w), int(bboxC.height * h)

                matched_name = "Unknown"

                if scan_encodings and known_encodings:
                    # Compare current face against known faces
                    matches = face_recognition.compare_faces(known_encodings, scan_encodings[0], tolerance=0.6)
                    face_distances = face_recognition.face_distance(known_encodings, scan_encodings[0])
                    
                    if True in matches:
                        best_match_index = np.argmin(face_distances)
                        if matches[best_match_index]:
                            matched_name = known_names[best_match_index]

                # Visual Feedback
                if matched_name != "Unknown":
                    color = (0, 255, 0) # Green for verified
                    log_attendance(matched_name)
                    st.success(f"✅ Access Granted: **{matched_name}**")
                else:
                    color = (0, 0, 255) # Red for unknown
                    st.error("❌ Access Denied: Unrecognized Face")

                # Draw Bounding Box & Label
                cv2.rectangle(frame, (xmin, ymin), (xmin + box_w, ymin + box_h), color, 2)
                cv2.putText(frame, matched_name, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption="Gate Scanner Output")

# =========================================================
# TAB 3: LOGS
# =========================================================
with tab3:
    st.header("Attendance Records")
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r") as f:
            st.code(f.read(), language="csv")