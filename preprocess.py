import cv2
import os
import glob
import csv
import random

# Settings
CLASSES = ["dancing", "fighting", "walking"]
FPS = 2
IMG_SIZE = 224
BASE_DIR = "E:/action_detection"
MIN_FILE_SIZE_KB = 50

def extract_frames(video_path, output_dir):
    if os.path.exists(output_dir) and len(os.listdir(output_dir)) > 0:
        return len(os.listdir(output_dir))
    if os.path.getsize(video_path) < MIN_FILE_SIZE_KB * 1024:
        print(f"  Skipping {os.path.basename(video_path)} - too small")
        return 0
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps <= 0:
        print(f"  Skipping - could not read FPS")
        cap.release()
        return 0
    interval = max(1, int(video_fps / FPS))
    frame_count, saved = 0, 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % interval == 0:
            frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
            cv2.imwrite(f"{output_dir}/frame_{saved:04d}.jpg", frame)
            saved += 1
        frame_count += 1
    cap.release()
    return saved

def process_all():
    all_clips = []
    for class_name in CLASSES:
        video_folder = f"{BASE_DIR}/data/raw/{class_name}"
        videos = glob.glob(f"{video_folder}/*.mp4")
        print(f"\n{class_name}: {len(videos)} videos found")
        for video_path in videos:
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            video_name = "".join(
                c if c.isalnum() or c == "_" else "_"
                for c in video_name
            )
            output_dir = f"{BASE_DIR}/data/frames/{class_name}/{video_name}"
            saved = extract_frames(video_path, output_dir)
            if saved > 0:
                all_clips.append([output_dir, class_name])
                print(f"  {video_name}: {saved} frames saved")
    return all_clips

def create_splits(all_clips):
    random.seed(42)
    random.shuffle(all_clips)
    total = len(all_clips)
    train_end = int(total * 0.70)
    val_end   = int(total * 0.85)
    splits = {
        "train": all_clips[:train_end],
        "val":   all_clips[train_end:val_end],
        "test":  all_clips[val_end:]
    }
    for split_name, split_data in splits.items():
        path = f"{BASE_DIR}/data/{split_name}.csv"
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["clip_path", "label"])
            writer.writerows(split_data)
        print(f"{split_name}: {len(split_data)} clips")

if __name__ == "__main__":
    print("=" * 50)
    print("Starting/Resuming preprocessing...")
    print("=" * 50)
    all_clips = process_all()
    print("\n" + "=" * 50)
    print(f"Total clips available: {len(all_clips)}")
    print("=" * 50)
    if len(all_clips) == 0:
        print("No clips found! Check your data/raw folders.")
    else:
        create_splits(all_clips)
        print("\nPreprocessing complete!")