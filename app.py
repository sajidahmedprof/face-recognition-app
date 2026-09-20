import os
import cv2
import numpy as np
import face_recognition
import streamlit as st
from datetime import datetime

# ==========================================
# PAGE CONFIGURATION & STYLING
# ==========================================
st.set_page_config(
    page_title="AI Attendance & Face Recognition System",
    page_icon="🪪",
    layout="wide"
)

FACES_DIR = "faces"
LOG_FILE = "attendance.csv"

# Initialize required storage directories and files
if not os.path.exists(FACES_DIR):
    os.makedirs(FACES_DIR)

if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w") as f:
        f.write("Name,Date,Time\n")

# ==========================================
# CORE ENGINES & HELPER FUNCTIONS
# ==========================================
@st.cache_data(ttl=5)  # Auto-refresh profile cache every 5 seconds
def load_registered_faces():
    """
    Scans the faces/ directory, loads image files, and extracts
    128-dimensional facial vector embeddings for all users.
    """
    known_encodings = []
    known_names = []
    
    for file in os.listdir(FACES_DIR):
        if file.lower().endswith((".jpg", ".jpeg", ".png")):
            name = os.path.splitext(file)[0].replace("_", " ")
            img_path = os.path.join(FACES_DIR, file)
            try:
                img = face_recognition.load_image_file(img_path)
                encodings = face_recognition.face_encodings(img)
                if len(encodings) > 0:
                    known_encodings.append(encodings[0])
                    known_names.append(name)
            except Exception as e:
                pass
                
    return known_encodings, known_names

def log_attendance(name):
    """
    Logs recognized employees into attendance.csv with exact timestamps.
    Includes duplicate prevention logic for the same day/session.
    """
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")

    # Prevent spamming duplicate logs in the file
    already_logged = False
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r") as f:
            lines = f.readlines()
            for line in reversed(lines):
                parts = line.strip().split(",")
                if len(parts) == 3:
                    if parts[0] == name and parts[1] == date_str:
                        already_logged = True
                        break

    if not already_logged:
        with open(LOG_FILE, "a") as f:
            f.write(f"{name},{date_str},{time_str}\n")
        return True
    return False

# ==========================================
# USER INTERFACE & NAVIGATION
# ==========================================
st.title("🛡️ Smart Office AI Face Recognition & Self-Registration System")
st.caption("Powered by OpenCV, dlib face-recognition vector embeddings, and Streamlit")

tab1, tab2, tab3 = st.tabs([
    "📸 Self-Registration (Add Face)", 
    "🔍 Live Gate Scanner", 
    "📊 Attendance Logs"
])

# ------------------------------------------
# TAB 1: SELF REGISTRATION MODULE
# ------------------------------------------
with tab1:
    st.header("Employee Self-Registration Portal")
    st.write("Register yourself by entering your full name and taking a clear webcam snapshot.")

    col1, col2 = st.columns([1, 1])

    with col1:
        user_name = st.text_input("Full Name:", placeholder="e.g. Alice Smith")
        img_buffer = st.camera_input("Capture Profile Photo", key="register_cam")

    with col2:
        st.info("📌 **Guidelines for Registration:**\n"
                "- Face the camera directly in good lighting.\n"
                "- Ensure only ONE person is visible in the frame.\n"
                "- Remove heavy masks or dark sunglasses.")

        if st.button("Save Face Profile", type="primary", use_container_width=True):
            if not user_name.strip():
                st.error("⚠️ Please enter your full name before saving.")
            elif img_buffer is None:
                st.error("⚠️ Please capture a photo using the webcam box on the left.")
            else:
                # Decode uploaded image buffer to OpenCV RGB image
                bytes_data = img_buffer.getvalue()
                cv_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
                rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)

                # Detect face locations & validate single face input
                locations = face_recognition.face_locations(rgb_img)

                if len(locations) == 0:
                    st.error("❌ No face detected! Please adjust your position and retake.")
                elif len(locations) > 1:
                    st.warning("⚠️ Multiple faces detected! Make sure only you are in the frame.")
                else:
                    # Save image using sanitized filename
                    clean_name = user_name.strip().replace(" ", "_")
                    file_path = os.path.join(FACES_DIR, f"{clean_name}.jpg")
                    cv2.imwrite(file_path, cv_img)

                    st.success(f"✅ Profile successfully registered for **{user_name}**!")
                    st.balloons()
                    st.cache_data.clear()  # Refresh cached encodings immediately

# ------------------------------------------
# TAB 2: REAL-TIME GATE SCANNER MODULE
# ------------------------------------------
with tab2:
    st.header("Live Verification Gate")
    st.write("Scan your face at the entrance gate to verify identity and mark attendance.")

    scan_buffer = st.camera_input("Scan Face for Access", key="gate_cam")

    if scan_buffer is not None:
        known_encodings, known_names = load_registered_faces()

        # Convert buffer to OpenCV frame
        bytes_data = scan_buffer.getvalue()
        frame = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        
        # Downscale frame to 25% size for faster vector extraction speed
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # Detect face locations & compute 128-d embeddings
        locations = face_recognition.face_locations(rgb_small_frame)
        encodings = face_recognition.face_encodings(rgb_small_frame, locations)

        if len(locations) == 0:
            st.warning("⚠️ No face detected in camera view.")
        else:
            for encoding, loc in zip(encodings, locations):
                name = "Unknown"

                if len(known_encodings) > 0:
                    # Compute L2 Euclidean distance metric
                    distances = face_recognition.face_distance(known_encodings, encoding)
                    best_match_idx = np.argmin(distances)

                    # Distance tolerance threshold (lower is stricter)
                    if distances[best_match_idx] <= 0.45:
                        name = known_names[best_match_idx]

                # Scale back face bounding box coordinates (1/0.25 = 4x)
                top, right, bottom, left = [v * 4 for v in loc]

                if name != "Unknown":
                    color = (0, 255, 0)  # Green for Verified
                    new_entry = log_attendance(name)
                    if new_entry:
                        st.success(f"✅ **ACCESS GRANTED:** Welcome, **{name}**! Attendance recorded.")
                    else:
                        st.info(f"ℹ️ **ACCESS GRANTED:** Welcome back, **{name}**. (Already logged today).")
                else:
                    color = (0, 0, 255)  # Red for Unknown
                    st.error("❌ **ACCESS DENIED:** User identity not recognized!")

                # Draw UI overlay bounding box and text on the frame
                cv2.rectangle(frame, (left, top), (right, bottom), color, 3)
                cv2.rectangle(frame, (left, top - 35), (right, top), color, cv2.FILLED)
                cv2.putText(
                    frame, name, (left + 6, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2
                )

            # Display processed output in dashboard
            st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption="Gate Scanner Output")

# ------------------------------------------
# TAB 3: ATTENDANCE LOGS & ANALYTICS
# ------------------------------------------
with tab3:
    st.header("Recorded Office Logs")

    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r") as f:
            log_data = f.read()

        st.download_button(
            label="📥 Download CSV Log",
            data=log_data,
            file_name=f"attendance_log_{datetime.now().strftime('%Y_%m_%d')}.csv",
            mime="text/csv"
        )
        
        st.code(log_data, language="csv")
    else:
        st.info("No attendance records logged yet.")