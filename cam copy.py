import cv2
import torch
import numpy as np
import torchvision.transforms as transforms
from pytorch_i3d import InceptionI3d
import os
import time

# ========== Config ==========
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_CLASSES = 2000
MODEL_PATH = 'checkpoints/nslt_2000_065846_0.447803.pt'
LABELS_PATH = 'preprocess/wlasl_class_list.txt'
FRAME_WINDOW = 64  # Number of frames to buffer before prediction
DISPLAY_DELAY = 2  # Seconds to hold prediction on screen

# ========== Load Labels ==========
labels = [''] * NUM_CLASSES
if os.path.exists(LABELS_PATH):
    with open(LABELS_PATH, 'r') as f:
        for line in f:
            idx, label = line.strip().split('\t')
            labels[int(idx)] = label
else:
    labels = [f'Class {i}' for i in range(NUM_CLASSES)]

# ========== Load Model ==========
model = InceptionI3d(num_classes=NUM_CLASSES, in_channels=3)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# ========== Frame Preprocessing ==========
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# ========== Webcam Setup ==========
cap = cv2.VideoCapture(0)
frames = []

print("📷 Live Sign Language Recognition Started. Press 'q' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    tensor_frame = transform(rgb_frame)
    frames.append(tensor_frame)

    if len(frames) == FRAME_WINDOW:
        # Prepare input
        clip = torch.stack(frames)  # [T, C, H, W]
        clip = clip.permute(1, 0, 2, 3).unsqueeze(0).to(DEVICE)  # [1, 3, T, H, W]

        # Predict
        with torch.no_grad():
            logits = model(clip)  # [1, NUM_CLASSES, T']
            logits = torch.mean(logits, dim=2)  # Average over time
            pred_class = torch.argmax(logits, dim=1).item()
            pred_label = labels[pred_class] if labels else str(pred_class)

        print(f"🧠 Predicted Sign: {pred_label}")

        # Display prediction on frame
        overlay_frame = frame.copy()
        cv2.putText(overlay_frame, f'Prediction: {pred_label}', (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Show prediction for DISPLAY_DELAY seconds
        start_time = time.time()
        while time.time() - start_time < DISPLAY_DELAY:
            cv2.imshow("Sign Recognition", overlay_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        frames = []  # Clear frame buffer after prediction

    # Live feed window
    cv2.imshow("Sign Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ========== Cleanup ==========
cap.release()
cv2.destroyAllWindows()
print("👋 Exiting...")
