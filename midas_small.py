import cv2
import torch
import urllib.request
import os
import numpy as np
import time

# Download MiDaS model if not already
model_path = "weights/model-small.onnx"
if not os.path.exists(model_path):
    print("Downloading MiDaS model...")
    os.makedirs("weights", exist_ok=True)
    url = "https://github.com/isl-org/MiDaS/releases/download/v2_1/model-small.onnx"

    urllib.request.urlretrieve(url, model_path)
    print("Download complete!")

# Load MiDaS small model (ONNX version for simplicity and speed)
model = cv2.dnn.readNet(model_path)
model.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
model.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# Webcam capture
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Webcam not accessible.")
    exit()

print("Press 'q' to quit.")
while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    input_blob = cv2.dnn.blobFromImage(frame, 1.0 / 255.0, (256, 256),
                                       mean=[0.485, 0.456, 0.406],
                                       swapRB=True, crop=False)
    model.setInput(input_blob)
    start = time.time()
    depth_map = model.forward()[0]
    end = time.time()

    # Resize depth to original size
    depth_map = cv2.resize(depth_map, (w, h))
    depth_map = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
    depth_map = depth_map.astype(np.uint8)
    depth_color = cv2.applyColorMap(depth_map, cv2.COLORMAP_INFERNO)

    # Combine original + depth map side by side
    combined = np.hstack((frame, depth_color))

    fps = f"FPS: {1/(end-start):.2f}"
    cv2.putText(combined, fps, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

    cv2.imshow("MiDaS Depth Estimation", combined)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()