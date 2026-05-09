import clip
import torch
import torch.nn as nn
import csv
import os
import glob
import random
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# Settings
CLASSES = ["dancing", "fighting", "walking"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
BASE_DIR = "E:/action_detection"
EPOCHS = 10
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {DEVICE}")

# Dataset
class ActionDataset(Dataset):
    def __init__(self, csv_path, preprocess):
        self.samples = []
        self.preprocess = preprocess
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                frames = glob.glob(f"{row['clip_path']}/*.jpg")
                if frames:
                    # use middle frame
                    frame = sorted(frames)[len(frames)//2]
                    self.samples.append((frame, CLASS_TO_IDX[row["label"]]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        frame_path, label = self.samples[idx]
        try:
            image = self.preprocess(Image.open(frame_path).convert("RGB"))
        except:
            image = torch.zeros(3, 224, 224)
        return image, label

# Model
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
        # Freeze CLIP weights
        for param in self.clip_model.parameters():
            param.requires_grad = False

    def forward(self, images):
        with torch.no_grad():
            features = self.clip_model.encode_image(images).float()
        return self.classifier(features)

def train():
    # Load CLIP
    print("Loading CLIP model...")
    clip_model, preprocess = clip.load("ViT-B/32", device=DEVICE)

    # Datasets
    print("Loading datasets...")
    train_dataset = ActionDataset(f"{BASE_DIR}/data/train.csv", preprocess)
    val_dataset   = ActionDataset(f"{BASE_DIR}/data/val.csv", preprocess)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples:   {len(val_dataset)}")

    # Model
    model = CLIPClassifier(clip_model, num_classes=len(CLASSES)).to(DEVICE)
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

    best_val_acc = 0.0

    for epoch in range(EPOCHS):
        # Training
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            preds = outputs.argmax(dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

        # Validation
        model.eval()
        val_correct, val_total = 0, 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                preds = outputs.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        train_acc = train_correct / train_total * 100
        val_acc   = val_correct / val_total * 100

        print(f"Epoch {epoch+1}/{EPOCHS} | "
              f"Loss: {train_loss/len(train_loader):.3f} | "
              f"Train Acc: {train_acc:.1f}% | "
              f"Val Acc: {val_acc:.1f}%")

        scheduler.step()

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs(f"{BASE_DIR}/models", exist_ok=True)
            torch.save(model.state_dict(), f"{BASE_DIR}/models/best_model.pth")
            print(f"  ✓ Best model saved! Val Acc: {val_acc:.1f}%")

    print(f"\nTraining complete! Best Val Accuracy: {best_val_acc:.1f}%")
    print(f"Model saved to: {BASE_DIR}/models/best_model.pth")

if __name__ == "__main__":
    train()