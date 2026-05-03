import streamlit as st
import cv2
import pandas as pd
import time
from backend.detector import process_frame
from backend.analytics import save_analytics

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Workspace Occupancy Detection",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
.main {
    background-color: #0E1117;
}

h1 {
    text-align: center;
    color: white;
    font-size: 42px;
}

div[data-testid="metric-container"] {
    background-color: #1E1E1E;
    border: 1px solid #333;
    padding: 15px;
    border-radius: 12px;
    text-align: center;
    box-shadow: 0px 0px 10px rgba(0,255,100,0.1);
}

.stButton>button {
    width: 100%;
    border-radius: 10px;
    background-color: #00C853;
    color: white;
    font-weight: bold;
    padding: 10px;
}

.stRadio > div {
    flex-direction: row;
}
</style>
""", unsafe_allow_html=True)

st.title("Workspace Occupancy Detection Dashboard")

st.sidebar.title("Settings")

confidence = st.sidebar.slider(
    "Detection Confidence",
    0.1,
    1.0,
    0.5
)

grid_size = st.sidebar.selectbox(
    "Grid Layout",
    ["3x3", "4x4", "5x5"]
)
# ---------------- SESSION STATE ----------------
if "webcam_running" not in st.session_state:
    st.session_state.webcam_running = False

if "last_saved" not in st.session_state:
    st.session_state.last_saved = time.time()
# ---------------- INPUT OPTION ----------------
option = st.radio("Choose Input:", ["Webcam", "Upload Video"])

# ---------------- PLACEHOLDERS ----------------
frame_placeholder = st.empty()

col1, col2, col3 = st.columns(3)

metric1 = col1.empty()
metric2 = col2.empty()
metric3 = col3.empty()


# ---------------- WEBCAM MODE ----------------
if option == "Webcam":

    btn1, btn2 = st.columns(2)

    if btn1.button("▶ Start Webcam"):
        st.session_state.webcam_running = True

    if btn2.button("⏹ Stop Webcam"):
        st.session_state.webcam_running = False

    if st.session_state.webcam_running:

        cap = cv2.VideoCapture(0)

        while st.session_state.webcam_running:

            ret, frame = cap.read()

            if not ret:
                st.error("Unable to access webcam.")
                break

            frame = cv2.resize(frame, (800, 600))

            processed_frame, pc, op, dz = process_frame(frame)

            current_time = time.time()

            if current_time - st.session_state.last_saved >= 5:
                save_analytics(pc, op, dz)
                st.session_state.last_saved = current_time

            processed_frame = cv2.cvtColor(
                processed_frame,
                cv2.COLOR_BGR2RGB
            )

            frame_placeholder.image(
                processed_frame,
                channels="RGB"
            )

            metric1.metric("People Count", pc)
            metric2.metric("Occupancy %", f"{op}%")
            metric3.metric(
                "Zones Active",
                ", ".join(dz) if dz else "None"
            )

        cap.release()


# ---------------- VIDEO MODE ----------------
elif option == "Upload Video":

    uploaded_file = st.file_uploader(
        "Upload a Video",
        type=["mp4", "avi", "mov"]
    )

    if uploaded_file:

        with open("temp_video.mp4", "wb") as f:
            f.write(uploaded_file.read())

        cap = cv2.VideoCapture("temp_video.mp4")

        while cap.isOpened():

            ret, frame = cap.read()

            if not ret:
                break

            frame = cv2.resize(frame, (800, 600))

            processed_frame, pc, op, dz = process_frame(frame)

            current_time = time.time()

            if current_time - st.session_state.last_saved >= 5:
                save_analytics(pc, op, dz)
                st.session_state.last_saved = current_time

            processed_frame = cv2.cvtColor(
                processed_frame,
                cv2.COLOR_BGR2RGB
            )

            frame_placeholder.image(
                processed_frame,
                channels="RGB"
            )

            metric1.metric("People Count", pc)
            metric2.metric("Occupancy %", f"{op}%")
            metric3.metric(
                "Zones Active",
                ", ".join(dz) if dz else "None"
            )

        cap.release()

# ---------------- ANALYTICS GRAPH ----------------
st.subheader("Analytics Overview")

try:
    df = pd.read_csv(
        "data/analytics.csv",
        header=None
    )

    df.columns = [
        "People Count",
        "Occupancy %",
        "Zones Active"
    ]

    st.line_chart(df["Occupancy %"])

except Exception as e:
    st.error(f"Analytics Error: {e}")


# python -m streamlit run frontend/app.py

