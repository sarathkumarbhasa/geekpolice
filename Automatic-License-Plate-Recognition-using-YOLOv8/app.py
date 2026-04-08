import streamlit as st
import os
import sys
import importlib

# Streamlit Cloud deployment patch for OpenCV
try:
    import cv2
except ImportError:
    os.system(f"{sys.executable} -m pip uninstall -y opencv-python opencv-python-headless")
    os.system(f"{sys.executable} -m pip install opencv-python-headless")
    importlib.invalidate_caches()
    if 'cv2' in sys.modules:
        del sys.modules['cv2']
    
import cv2
import numpy as np
import tempfile
import time
from datetime import datetime
import torch

from ultralytics import YOLO
from sort.sort import Sort
from util import get_car, read_license_plate, write_csv, get_region

st.set_page_config(page_title="ANPR Pro Surveillance", page_icon="🚓", layout="wide")

st.markdown("""
<style>
/* Premium Dark Mode Styling for Streamlit */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(to right, #000428, #004e92);
    color: white;
}
[data-testid="stSidebar"] {
    background-color: rgba(0, 0, 0, 0.6);
    border-right: 1px solid rgba(255, 255, 255, 0.1);
}
h1, h2, h3, h4, h5, h6, p, label, .stMarkdown {
    color: #e0e0e0 !important;
}
.stButton>button {
    background: linear-gradient(90deg, #0cebeb, #20e3b2, #29ffc6);
    color: #000;
    border: none;
    border-radius: 8px;
    padding: 10px 24px;
    font-weight: bold;
    transition: all 0.3s ease;
}
.stButton>button:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 15px rgba(32, 227, 178, 0.4);
    color: #fff;
}
.metric-container {
    background: rgba(255,255,255,0.05);
    border-radius: 10px;
    padding: 15px;
    border: 1px solid rgba(255,255,255,0.1);
    margin-bottom: 10px;
}
.glassmorphism {
    background: rgba(255, 255, 255, 0.05);
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
    border-radius: 10px;
    border: 1px solid rgba(255, 255, 255, 0.18);
    padding: 20px;
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    # To prevent weight loading issues on some versions of torch
    _original_load = torch.load
    def safe_load(*args, **kwargs):
        kwargs['weights_only'] = False
        return _original_load(*args, **kwargs)
    torch.load = safe_load
    
    coco_model = YOLO('yolov8n.pt')
    license_plate_detector = YOLO('license_plate_detector.pt')
    return coco_model, license_plate_detector

coco_model, license_plate_detector = load_models()
vehicles = [2, 3, 5, 7] # classes for car, motorcycle, bus, truck

def process_frame(frame, frame_nmr, mot_tracker, results_dict):
    detections = coco_model(frame, conf=0.1)[0]
    detections_ = []
    for detection in detections.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = detection
        if int(class_id) in vehicles:
            detections_.append([x1, y1, x2, y2, score])

    track_ids = mot_tracker.update(np.asarray(detections_) if len(detections_) > 0 else np.empty((0, 5)))

    license_plates = license_plate_detector(frame, conf=0.05)[0]

    for i, license_plate in enumerate(license_plates.boxes.data.tolist()):
        x1, y1, x2, y2, score, class_id = license_plate
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)

        if car_id != -1:
            cv2.putText(frame, f"Car ID: {int(car_id)}", (int(xcar1), int(ycar1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        padding = 5
        x1_p = max(0, int(x1) - padding)
        y1_p = max(0, int(y1) - padding)
        x2_p = min(frame.shape[1], int(x2) + padding)
        y2_p = min(frame.shape[0], int(y2) + padding)
        license_plate_crop = frame[y1_p:y2_p, x1_p:x2_p, :]

        license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop)

        if license_plate_text is not None:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            region = get_region(license_plate_text)
            cv2.putText(frame, f"{license_plate_text}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

            obj_id = car_id if car_id != -1 else f"Obj_{frame_nmr}_{i}"

            res_entry = {
                'license_plate': {'bbox': [x1, y1, x2, y2],
                                  'text': license_plate_text,
                                  'bbox_score': score,
                                  'text_score': license_plate_text_score},
                'timestamp': timestamp,
                'lane': "N/A",
                'speed': '0',
                'region': region
            }
            if car_id != -1:
                res_entry['car'] = {'bbox': [xcar1, ycar1, xcar2, ycar2]}
            
            results_dict.setdefault(frame_nmr, {})[obj_id] = res_entry

    return frame

def main():
    st.markdown("<h1 style='text-align: center; font-size: 3rem;'>🚓 ANPR Pro Surveillance Dashboard</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size: 1.2rem; color: #a1a1aa;'>Real-time AI-powered Automatic License Plate Recognition System</p>", unsafe_allow_html=True)
    st.markdown("<hr style='border:1px solid #ffffff22'>", unsafe_allow_html=True)

    st.sidebar.markdown("<h2 style='text-align: center;'>🎛️ Control Panel</h2>", unsafe_allow_html=True)
    st.sidebar.markdown("<hr style='border:1px solid #ffffff22'>", unsafe_allow_html=True)
    input_mode = st.sidebar.radio("📡 Select Input Source", ["Upload Video", "Live Camera"])

    results = {}
    mot_tracker = Sort()

    col1, col2 = st.columns([3, 1])

    with col1:
        st.markdown("<div class='glassmorphism' style='margin-bottom:10px;'><h3>📹 Live Feed</h3></div>", unsafe_allow_html=True)
        frame_placeholder = st.empty()

    with col2:
        st.markdown("<div class='glassmorphism' style='margin-bottom:10px;'><h3>📊 Status</h3></div>", unsafe_allow_html=True)
        status_placeholder = st.empty()

    if input_mode == "Upload Video":
        st.sidebar.markdown("### 📂 Video Upload")
        uploaded_file = st.sidebar.file_uploader("Select an MP4/AVI file", type=["mp4", "avi", "mov"])
        
        if uploaded_file is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_file.read())
            
            cap = cv2.VideoCapture(tfile.name)
            st.sidebar.success("✅ Media uploaded successfully.")
            
            start_btn = st.sidebar.button("▶️ Process Video")
            stop_btn = st.sidebar.button("🛑 Stop Processing")
            
            if start_btn:
                status_placeholder.markdown("<div class='metric-container'>Engine: <b style='color:#0cebeb'>Processing Data</b></div>", unsafe_allow_html=True)
                frame_nmr = -1
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret or stop_btn:
                        break
                        
                    frame_nmr += 1
                    processed_frame = process_frame(frame, frame_nmr, mot_tracker, results)
                    
                    processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                    frame_placeholder.image(processed_frame, channels="RGB", use_column_width=True)
                    
                cap.release()
                write_csv(results, './results.csv')
                status_placeholder.markdown("<div class='metric-container' style='border-color: #00ff00;'>Engine: <b style='color:#00ff00'>Completed</b><br/>Saved: results.csv</div>", unsafe_allow_html=True)
                st.balloons()
                
    elif input_mode == "Live Camera":
        st.sidebar.markdown("### 🎥 Camera Input")
        camera_index = st.sidebar.selectbox("Select Camera Index", [0, 1, 2], index=0)
        
        run_camera = st.sidebar.checkbox("🟢 Start Camera Live Stream")
        
        if run_camera:
            status_placeholder.markdown("<div class='metric-container'>Engine: <b style='color:#0cebeb'>Live Mode</b></div>", unsafe_allow_html=True)
            cap = cv2.VideoCapture(camera_index)
            frame_nmr = -1
            while cap.isOpened() and run_camera:
                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to capture frame from camera.")
                    break
                
                frame_nmr += 1
                processed_frame = process_frame(frame, frame_nmr, mot_tracker, results)
                
                processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(processed_frame, channels="RGB", use_column_width=True)
                
            cap.release()
            write_csv(results, './camera_results.csv')
            status_placeholder.markdown("<div class='metric-container' style='border-color: #00ff00;'>Engine: <b style='color:#ff0000'>Stopped</b><br/>Saved: camera_results.csv</div>", unsafe_allow_html=True)

if __name__ == '__main__':
    main()
