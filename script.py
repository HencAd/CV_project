from transformers import AutoImageProcessor, TimesformerForVideoClassification
import numpy as np
import torch
import cv2
from PIL import Image
from peft import PeftModel, PeftConfig
import random 
import torch.nn as nn

def read_mp4_to_pil_images(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    pil_images = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  
        pil_image = Image.fromarray(frame)
        pil_images.append(pil_image)

    cap.release()

    return pil_images, frame_rate, width, height

video_path = "dataset/train/commercial/20250127-200939_prosieben_001.mp4"
pil_images, frame_rate, width, height = read_mp4_to_pil_images(video_path)
selected_indices = [0, 35, 70, 105, 140, 175, 210, 235]  
selected_frames = [pil_images[i] for i in selected_indices if i < len(pil_images)]

processor = AutoImageProcessor.from_pretrained("facebook/timesformer-base-finetuned-k400")

lora_model_path = "final_model_test"
final_model = TimesformerForVideoClassification.from_pretrained(lora_model_path)

# Access the classifier layer
classifier = final_model.classifier
classifier_weights = classifier.weight
print(classifier_weights)

num_output_classes = final_model.classifier.out_features 

if num_output_classes == 2:
    print("The loaded adapter is trained for 2-class classification.")

final_model.eval()

inputs = processor(selected_frames, return_tensors="pt")

with torch.no_grad():
  outputs = final_model(**inputs)
  logits = outputs.logits
  print(f"Predicted probabilities: {torch.softmax(outputs.logits, dim=1)}")

predicted_class_idx = logits.argmax(-1).item()
print("Predicted class:", final_model.config.id2label[predicted_class_idx])
