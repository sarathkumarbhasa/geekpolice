import ast

import cv2
import numpy as np
import pandas as pd


def draw_border(img, top_left, bottom_right, color=(0, 255, 0), thickness=10, line_length_x=200, line_length_y=200):
    x1, y1 = top_left
    x2, y2 = bottom_right

    cv2.line(img, (x1, y1), (x1, y1 + line_length_y), color, thickness)  #-- top-left
    cv2.line(img, (x1, y1), (x1 + line_length_x, y1), color, thickness)

    cv2.line(img, (x1, y2), (x1, y2 - line_length_y), color, thickness)  #-- bottom-left
    cv2.line(img, (x1, y2), (x1 + line_length_x, y2), color, thickness)

    cv2.line(img, (x2, y1), (x2 - line_length_x, y1), color, thickness)  #-- top-right
    cv2.line(img, (x2, y1), (x2, y1 + line_length_y), color, thickness)

    cv2.line(img, (x2, y2), (x2, y2 - line_length_y), color, thickness)  #-- bottom-right
    cv2.line(img, (x2, y2), (x2 - line_length_x, y2), color, thickness)

    return img


results = pd.read_csv('./filter.csv')

# load video
video_path = 'sample.mp4'
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Error: Could not open video {video_path}")
    exit()

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter('./out.mp4', fourcc, fps, (width, height))

# We'll use a simplified visualization since the CSV no longer has bboxes
# We'll draw the detected info at the top of the frame for each detected car
frame_nmr = -1
while True:
    ret, frame = cap.read()
    frame_nmr += 1
    if not ret:
        break

    # For visualization, we'll just show the last detected plate info on screen
    # since we don't have bboxes to draw on specific cars anymore
    current_data = results.head(5) # Show some recent data as an example overlay
    
    y_offset = 50
    for i, row in current_data.iterrows():
        info_text = f"ID: {row['car_id']} | Plate: {row['license_number']} | {row['region']} | {row['lane']}"
        cv2.putText(frame, info_text, (50, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        y_offset += 40

    out.write(frame)

out.release()
cap.release()
print("Visualization saved to out.mp4")
