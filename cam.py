import cv2
import torch
import numpy as np
import torchvision.transforms as transforms
from pytorch_i3d import InceptionI3d
import os

# ======= Setup ========
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_CLASSES = 2000  # Update as per your model
MODEL_PATH = 'checkpoints/nslt_2000_065846_0.447803.pt'
LABELS_PATH = 'preprocess/wlasl_class_list.txt'

labels = [''] * NUM_CLASSES


# ======= Load Model ========
model = InceptionI3d(num_classes=NUM_CLASSES, in_channels=3)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# ======= Load Labels (Optional) ========
if os.path.exists(LABELS_PATH):
    with open(LABELS_PATH, 'r') as f:
        for line in f:
            idx, label = line.strip().split('\t')
            labels[int(idx)] = label
else:
    labels = [f'Class {i}' for i in range(NUM_CLASSES)]

# ======= Video Capture and Transform ========
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

cap = cv2.VideoCapture(0)
frames = []
frame_window = 64  # Number of frames per prediction

print("🎥 Live camera started. Press 'q' to exit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_tensor = transform(rgb_frame)
    frames.append(img_tensor)

    if len(frames) == frame_window:
        clip = torch.stack(frames)  # [T, C, H, W]
        clip = clip.permute(1, 0, 2, 3).unsqueeze(0).to(DEVICE)  # [1, 3, T, H, W]

        with torch.no_grad():
            logits = model(clip)  # [1, NUM_CLASSES, T']
            logits = torch.mean(logits, dim=2)  # Average over time dimension → [1, NUM_CLASSES]

        pred_class = torch.argmax(logits, dim=1).item()
        pred_label = labels[pred_class] if labels else str(pred_class)
        print(f"🧠 Predicted Sign: {pred_label}")

        frames = []  # Reset for next prediction

    cv2.imshow("Sign Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
