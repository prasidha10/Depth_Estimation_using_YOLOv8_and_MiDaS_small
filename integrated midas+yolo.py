import cv2
import torch
import urllib.request
import os
import numpy as np
from ultralytics import YOLO

# Download MiDaS model if not already present
model_path = "weights/model-small.onnx"
if not os.path.exists(model_path):
    print("Downloading MiDaS model...")
    os.makedirs("weights", exist_ok=True)
    url = "https://github.com/isl-org/MiDaS/releases/download/v2_1/model-small.onnx"
    urllib.request.urlretrieve(url, model_path)
    print("Download complete!")

# Load MiDaS model
midas_model = cv2.dnn.readNet(model_path)
midas_model.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
midas_model.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# Load YOLOv8 model
model = YOLO('yolov8s.pt')  # Load YOLOv8s weights

# Start webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Webcam not accessible.")
    exit()

print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    height, width, _ = frame.shape

    # ==== Run MiDaS Depth Estimation ====
    blob = cv2.dnn.blobFromImage(frame, 1.0 / 255.0, (256, 256),
                                 mean=[0.485, 0.456, 0.406],
                                 swapRB=True, crop=False)
    midas_model.setInput(blob)
    depth_map = midas_model.forward()[0]
    depth_map = cv2.resize(depth_map, (width, height))
    depth_map_norm = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
    depth_map_color = cv2.applyColorMap(depth_map_norm.astype(np.uint8), cv2.COLORMAP_INFERNO)

    # ==== Run YOLOv8 ====
    results = model(frame)[0]  # Get the first result

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])
        class_name = model.names[cls_id]

        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
        if 0 <= cx < width and 0 <= cy < height:
            z = int(depth_map[cy, cx])
        else:
            z = 0

        # ==== Draw 3D Cube (Cuboid) ====
        box_width = x2 - x1
        box_height = y2 - y1
        scale = 1000 / (z + 100)
        offset_x = int(box_width * 0.3 * scale)
        offset_y = int(box_height * 0.3 * scale)

        pt1, pt2, pt3, pt4 = (x1, y1), (x2, y1), (x2, y2), (x1, y2)
        bpt1 = (x1 + offset_x, y1 + offset_y)
        bpt2 = (x2 + offset_x, y1 + offset_y)
        bpt3 = (x2 + offset_x, y2 + offset_y)
        bpt4 = (x1 + offset_x, y2 + offset_y)

        color = (0, int(z * 2) % 255, 255 - int(z) % 255)

        # Draw cuboid
        for a, b in [(pt1, pt2), (pt2, pt3), (pt3, pt4), (pt4, pt1),
                     (bpt1, bpt2), (bpt2, bpt3), (bpt3, bpt4), (bpt4, bpt1),
                     (pt1, bpt1), (pt2, bpt2), (pt3, bpt3), (pt4, bpt4)]:
            cv2.line(frame, a, b, color, 2)

        label = f"{class_name} ({cx},{cy},{z}) - Conf: {conf:.2f}"
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Combine original frame with depth map
    combined_display = np.hstack((frame, depth_map_color))

    cv2.imshow("YOLO + MiDaS 3D Cubes + Depth", combined_display)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()