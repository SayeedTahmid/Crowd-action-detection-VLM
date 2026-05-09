import clip
import torch
import torch.nn as nn
import csv
import glob
from PIL import Image
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

CLASSES = ["dancing", "fighting", "walking"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
BASE_DIR = "E:/action_detection"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class CLIPClassifier(nn.Module):
    def __init__(self, clip_model, num_classes):
        super().__init__()
        self.clip_model = clip_model
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        for param in self.clip_model.parameters():
            param.requires_grad = False

    def forward(self, images):
        with torch.no_grad():
            features = self.clip_model.encode_image(images).float()
        return self.classifier(features)

# Load model
print("Loading model...")
clip_model, preprocess = clip.load("ViT-B/32", device=DEVICE)
model = CLIPClassifier(clip_model, num_classes=3).to(DEVICE)
model.load_state_dict(torch.load(f"{BASE_DIR}/models/best_model.pth"))
model.eval()

# Load test data
print("Running on test set...")
all_preds, all_labels = [], []

with open(f"{BASE_DIR}/data/test.csv", "r") as f:
    reader = csv.DictReader(f)
    rows = list(reader)

for row in rows:
    frames = glob.glob(f"{row['clip_path']}/*.jpg")
    if not frames:
        continue
    frame_path = sorted(frames)[len(frames)//2]
    try:
        image = preprocess(Image.open(frame_path).convert("RGB")).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            output = model(image)
            pred = output.argmax(dim=1).item()
        all_preds.append(pred)
        all_labels.append(CLASS_TO_IDX[row["label"]])
    except:
        continue

# Results
print("\n" + "="*50)
print("TEST SET RESULTS")
print("="*50)
print(classification_report(all_labels, all_preds, target_names=CLASSES))

# Confusion matrix
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=CLASSES, yticklabels=CLASSES)
plt.title("Confusion Matrix - Crowd Action Detection")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.tight_layout()
plt.savefig(f"{BASE_DIR}/results/confusion_matrix.png", dpi=150)
plt.show()
print(f"\nConfusion matrix saved to results/confusion_matrix.png")