import os
os.system("pip install git+https://github.com/openai/CLIP.git")

import clip
import torch
import torch.nn as nn
import gradio as gr
import cv2
import numpy as np
from PIL import Image
import tempfile

CLASSES = ["dancing", "fighting", "walking"]
BASE_DIR = "."
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

# Load model once at startup
print("Loading model...")
clip_model, preprocess = clip.load("ViT-B/32", device=DEVICE)
model = CLIPClassifier(clip_model, num_classes=3).to(DEVICE)
model.load_state_dict(torch.load(f"{BASE_DIR}/models/best_model.pth", map_location=torch.device('cpu')))
model.eval()
print("Model ready!")

def predict_video(video_path):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Sample 5 frames evenly
    predictions = []
    frame_indices = np.linspace(0, total_frames-1, 5, dtype=int)
    
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = preprocess(Image.fromarray(frame_rgb)).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            output = model(image)
            probs = torch.softmax(output, dim=1)[0]
            predictions.append(probs.cpu().numpy())
    
    cap.release()
    
    if not predictions:
        return {"Error": 1.0}
    
    # Average predictions across frames
    avg_probs = np.mean(predictions, axis=0)
    
    return {CLASSES[i]: float(avg_probs[i]) for i in range(len(CLASSES))}

# Build Gradio interface
demo = gr.Interface(
    fn=predict_video,
    inputs=gr.Video(label="Upload crowd video clip"),
    outputs=gr.Label(num_top_classes=3, label="Detected Action"),
    title="Crowd Action Detection",
    description="Upload a short video clip to detect crowd actions: Dancing, Fighting, or Walking",
    examples=[],
    theme=gr.themes.Soft()
)

demo.launch(share=True, max_file_size="500mb")