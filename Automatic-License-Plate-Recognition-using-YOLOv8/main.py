import torch
from datetime import datetime
_original_load = torch.load
def safe_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _original_load(*args, **kwargs)
torch.load = safe_load

from ultralytics import YOLO
import cv2
import numpy as np

from sort.sort import Sort
from util import get_car, read_license_plate, write_csv, get_region


results = {}

# load models
coco_model = YOLO('yolov8n.pt')
license_plate_detector = YOLO(r'C:\Users\sarat\OneDrive\Desktop\geekpolice\Automatic-License-Plate-Recognition-using-YOLOv8\license_plate_detector.pt')

# load video
cap = cv2.VideoCapture(r'C:\Users\sarat\OneDrive\Desktop\geekpolice\Automatic-License-Plate-Recognition-using-YOLOv8\sample.mp4')
fps = cap.get(cv2.CAP_PROP_FPS)
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)

mot_tracker = Sort()

vehicles = [2, 3, 5, 7]

# read frames
frame_nmr = -1
ret = True
while ret:
    frame_nmr += 1
    ret, frame = cap.read()
    if ret:
        results[frame_nmr] = {}
        # detect vehicles
        detections = coco_model(frame, conf=0.1)[0]
        detections_ = []
        for detection in detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = detection
            if int(class_id) in vehicles:
                detections_.append([x1, y1, x2, y2, score])

        # track vehicles
        track_ids = mot_tracker.update(np.asarray(detections_) if len(detections_) > 0 else np.empty((0, 5)))

        # detect license plates
        license_plates = license_plate_detector(frame, conf=0.05)[0]
        for license_plate in license_plates.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = license_plate

            # assign license plate to car
            xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)

            if car_id != -1:

                # crop license plate
                license_plate_crop = frame[int(y1):int(y2), int(x1): int(x2), :]

                # read license plate number
                # We pass the original color crop to EasyOCR for better deep learning performance
                license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop)

                if license_plate_text is not None:
                    # Get current system time for timestamp
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                    # Get region (Indian state) from license plate
                    region = get_region(license_plate_text)

                    # Estimate lane based on car center x-coordinate
                    car_center_x = (xcar1 + xcar2) / 2
                    if width > 0:
                        lane_num = int((car_center_x / width) * 3) + 1 # Assuming 3 lanes
                        lane = f"Lane {lane_num}"
                    else:
                        lane = "N/A"

                    results[frame_nmr][car_id] = {'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
                                                  'license_plate': {'bbox': [x1, y1, x2, y2],
                                                                    'text': license_plate_text,
                                                                    'bbox_score': score,
                                                                    'text_score': license_plate_text_score},
                                                  'timestamp': timestamp,
                                                  'lane': lane,
                                                  'speed': '0',
                                                  'region': region} # Speed estimation requires multi-frame tracking

# write results
write_csv(results, './test.csv')
