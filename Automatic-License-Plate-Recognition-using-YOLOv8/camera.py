import torch
from datetime import datetime
import cv2
import numpy as np
import signal
import sys
import time
from ultralytics import YOLO
from sort.sort import Sort
from util import get_car, read_license_plate, write_csv, get_region

def save_results():
    if results:
        count = sum(len(v) for v in results.values())
        print(f"Saving {count} detection(s) to camera.csv...")
        write_csv(results, './camera.csv')
    else:
        print("No detections were made. Results dictionary is empty.")

# Add signal handler to save CSV on Ctrl+C
def signal_handler(sig, frame):
    print('\nInterrupted. Attempting to save results...')
    save_results()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

# To prevent weight loading issues on some versions of torch
_original_load = torch.load
def safe_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _original_load(*args, **kwargs)
torch.load = safe_load

results = {}

# Load models
print("Loading models...")
coco_model = YOLO('yolov8n.pt')
license_plate_detector = YOLO(r'C:\Users\sarat\OneDrive\Desktop\geekpolice\Automatic-License-Plate-Recognition-using-YOLOv8\license_plate_detector.pt')

# Open Camera
print("Attempting to open camera...")
cap = None
# Try common indices and backends
for index in range(10): # Try 0 to 9
    for backend in [None, cv2.CAP_DSHOW, cv2.CAP_MSMF]:
        backend_name = "Default" if backend is None else ("DSHOW" if backend == cv2.CAP_DSHOW else "MSMF")
        print(f"Trying camera index {index} with {backend_name} backend...")
        
        if backend is None:
            cap = cv2.VideoCapture(index)
        else:
            cap = cv2.VideoCapture(index, backend)
            
        if cap.isOpened():
            # Wait a bit for the camera to initialize
            time.sleep(2)
            ret, frame = cap.read()
            if ret:
                print(f"Success: Camera found at index {index} ({backend_name}).")
                break
        
        if cap:
            cap.release()
            cap = None
    if cap and cap.isOpened():
        break

if cap is None or not cap.isOpened():
    print("\nERROR: No working camera found.")
    print("Troubleshooting steps:")
    print("1. Ensure your camera is plugged in.")
    print("2. Close other apps that might be using the camera (Zoom, Teams, Browser).")
    print("3. Check Windows Privacy Settings -> Camera -> 'Allow apps to access your camera'.")
    exit()

width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
mot_tracker = Sort()
vehicles = [2, 3, 5, 7] # classes for car, motorcycle, bus, truck

print("Starting camera feed. Press 'q' to stop and save results.")

frame_nmr = -1
while True:
    frame_nmr += 1
    ret, frame = cap.read()
    if ret:
        # Detect vehicles
        detections = coco_model(frame, conf=0.1)[0]
        detections_ = []
        for detection in detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = detection
            if int(class_id) in vehicles:
                detections_.append([x1, y1, x2, y2, score])
        
        if len(detections_) > 0:
            print(f"Frame {frame_nmr}: Detected {len(detections_)} vehicles.")

        # Track vehicles
        track_ids = mot_tracker.update(np.asarray(detections_) if len(detections_) > 0 else np.empty((0, 5)))

        # Detect license plates
        license_plates = license_plate_detector(frame, conf=0.05)[0]
        lp_count = len(license_plates.boxes.data.tolist())
        if lp_count > 0:
            print(f"Frame {frame_nmr}: Detected {lp_count} license plates.")

        for i, license_plate in enumerate(license_plates.boxes.data.tolist()):
            x1, y1, x2, y2, score, class_id = license_plate
            
            # Draw license plate region for visual feedback
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

            # Assign license plate to car
            xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)

            # Define unique ID for this detection (use car_id if available, else a frame-specific index)
            obj_id = car_id if car_id != -1 else f"Standalone_{frame_nmr}_{i}"

            if car_id != -1:
                # Draw car tracking ID for feedback
                cv2.putText(frame, f"Car ID: {car_id}", (int(xcar1), int(ycar1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            
            # Crop license plate with a bit of padding for better OCR
            padding = 5
            x1_p = max(0, int(x1) - padding)
            y1_p = max(0, int(y1) - padding)
            x2_p = min(frame.shape[1], int(x2) + padding)
            y2_p = min(frame.shape[0], int(y2) + padding)
            license_plate_crop = frame[y1_p:y2_p, x1_p:x2_p, :]

            # Read license plate number
            # We pass the original color crop to EasyOCR for better deep learning performance
            license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop)

            if license_plate_text is not None:
                # Get current system time for timestamp
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                # Get region (Indian state) from license plate
                region = get_region(license_plate_text)
                
                # Draw plate text for feedback
                cv2.putText(frame, f"{license_plate_text}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
                
                print(f"Frame {frame_nmr}: Car {car_id if car_id != -1 else 'N/A'} -> Plate '{license_plate_text}' ({region})")

                # Estimate lane based on car center x-coordinate
                # Use plate center if car is not detected
                center_x = (xcar1 + xcar2) / 2 if car_id != -1 else (x1 + x2) / 2
                if width > 0:
                    lane_num = int((center_x / width) * 3) + 1 # Assuming 3 lanes
                    lane = f"Lane {lane_num}"
                else:
                    lane = "N/A"

                # Store result
                res_entry = {
                    'license_plate': {'bbox': [x1, y1, x2, y2],
                                      'text': license_plate_text,
                                      'bbox_score': score,
                                      'text_score': license_plate_text_score},
                    'timestamp': timestamp,
                    'lane': lane,
                    'speed': '0',
                    'region': region
                }
                
                if car_id != -1:
                    res_entry['car'] = {'bbox': [xcar1, ycar1, xcar2, ycar2]}
                
                results.setdefault(frame_nmr, {})[obj_id] = res_entry
            else:
                if car_id != -1:
                    print(f"Frame {frame_nmr}: Car {car_id} -> Plate detected but OCR failed.")
        else:
            # Note: No car_id != -1 check needed here anymore as we process all license plates
            pass
    
    else:
        # End of video or camera feed
        break

    # Display the frame
    try:
        cv2.imshow('ALPR Camera Feed', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    except cv2.error:
        # If GUI is not available (e.g. headless environment or missing libraries),
        # we can still process the frames and save the CSV.
        print("Warning: GUI not supported in this environment. Processing will continue in the background.")
        print("To stop, please interrupt the script (Ctrl+C).")
        # Since we can't use waitKey, we'll just continue until the camera stops or is interrupted.
        pass

# Release resources
if cap:
    cap.release()
cv2.destroyAllWindows()

# Write results
save_results()
