import os
import torch
import numpy as np
import argparse
from transformers import TimesformerForVideoClassification, AutoImageProcessor
import cv2
from collections import defaultdict

""" 
The script classifies new video data into 2 classes: "commercial" or "content" using the model trained by us https://huggingface.co/cvproject/final_model

Example of use:
python classify_new_data.py --data_dir=<data_dir>
"""

parser = argparse.ArgumentParser(description="Processing named arguments.")
parser.add_argument("--data_dir",      type=str,   required=True, help="Path to the directory where the videos are placed")

args = parser.parse_args()

data_dir = args.data_dir

print(f"Used parameters: data_dir: {data_dir}.")

def extract_frames(video_path, num_frames=8, resize=(224, 224)):
    """Extract evenly spaced frames from the video file."""
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Handle short videos by duplicating last frame
    if total_frames < num_frames:
        frame_indices = np.linspace(0, total_frames - 1, total_frames, dtype=int).tolist()
        frame_indices += [total_frames - 1] * (num_frames - total_frames)  # Pad with last frame
    else:
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)# Select evenly spaced frame indices

    frames = []
    for frame_index in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)  # Seek to the right frame
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        frame = cv2.resize(frame, resize)  # Resize to model input size
        frames.append(frame)

    cap.release()
    return frames

# Load the feature extractor
processor = AutoImageProcessor.from_pretrained("facebook/timesformer-base-finetuned-k400")

final_model = TimesformerForVideoClassification.from_pretrained('cvproject/final_model')

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
final_model.to(device)
final_model.eval()

label_freq = defaultdict(int)
for file in os.listdir(data_dir):
    if file.endswith(".mp4"):
        video_path = os.path.join(data_dir, file)
        print(f"File: {file}")
        frames = extract_frames(video_path)
        inputs = processor(frames, return_tensors="pt").to(device)  # Preprocess
        pixel_values = inputs["pixel_values"].to(device)

        # Run model inference on the sample
        with torch.no_grad():
            try:
                outputs = final_model(**{"pixel_values": pixel_values})
                predicted_label = torch.argmax(outputs.logits, dim=1).cpu().item()
                probabilities = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]
                print(f"Probabilities: {probabilities}")

                # Map predicted label to category
                predicted_class = final_model.config.id2label[predicted_label]
                print(f"Predicted class: {predicted_class}")

                label_freq[predicted_class] += 1
            except (RuntimeError, OSError, ValueError) as e:
                print(f"Skipping file: {file} - broken video. Error: {e}")
    else:
        print(f"Skipping file: {file} - not mp4")

for pred_class, cnt in label_freq.items():
        print('Detections for class {}: {}\n'.format(pred_class, cnt))


