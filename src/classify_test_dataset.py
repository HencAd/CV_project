import os
import torch
import wandb
import numpy as np
import argparse
import cv2
from pathlib import Path
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, recall_score, classification_report
from transformers import TimesformerForVideoClassification, AutoImageProcessor
import matplotlib.pyplot as plt
from tqdm import tqdm
import PIL
from PIL import Image
import VideoDataset

""" 
The script that classifies test dataset into 2 classes: "commercial" or "content" using the model trained in train_commercial_content_classification.py

It calculates accuracy, precision, recall, f1-score and prints classification_report.
It also collects every 50th video's picture + labels (true and predicted) and calculated probabilities and prints it out to visualize the result.

Example of use:
python classify_test_dataset.py --dataset_dir=<dataset_dir> --model_dir=<model_dir>
"""

wandb.login()  # Authenticate with your W&B account

parser1 = argparse.ArgumentParser(description="Processing named arguments.")
parser1.add_argument("--dataset_dir",   type=str,   required=True, help="Path to the dataset directory")
parser1.add_argument("--model_dir",     type=str,   required=True, help="Path to the directory where there is the model we wanto to use")
parser1.add_argument("--batch_size",    type=int,   default=4,     help="batch_size")
parser1.add_argument("--wandb_project", type=str,                  help="if you want to log to W&B please provide the project name")
parser1.add_argument("--wandb_run_name",type=str,                  help="if you want to log to W&B please provide the run name")
parser1.add_argument("--logging_steps", type=int,   default=50,    help="how often the script should log during training")

args1 = parser1.parse_args()

dataset_dir = args1.dataset_dir
model_dir = args1.model_dir
batch_size = args1.batch_size
wandb_project = args1.wandb_project
wandb_run_name = args1.wandb_run_name
logging_steps = args1.logging_steps

print(f"Used parameters: dataset_dir: {dataset_dir}, models_dir: {model_dir}, batch_size: {batch_size}.")
categories = ['content','commercial']

for phase in ['test']:
    phase_path = os.path.join(dataset_dir, phase)
    if not os.path.exists(phase_path):
        raise ("Dataset need to contain directories: 'train', 'val' and 'test'.")
    
    for category in (categories):
        category_path = os.path.join(phase_path, category)
        if not os.path.exists(category_path):
            raise ("Each subfolder of dataset (train, val, test) needs to contain directories: 'commercial' and 'content'.")


use_wandb = 0
if (wandb_project and wandb_run_name):
    use_wandb = 1

    wandb.init(project=wandb_project, name=wandb_run_name )


# Create datasets
path_test=os.path.join(dataset_dir, 'test')
# Load the feature extractor
feature_extractor = AutoImageProcessor.from_pretrained("facebook/timesformer-base-finetuned-k400")

# Create dataset
test_dataset = VideoDataset.VideoDataset(video_dir=path_test, feature_extractor=feature_extractor)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

final_model = TimesformerForVideoClassification.from_pretrained(model_dir)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
final_model.to(device)
final_model.eval()

vis_predicted_classes = []
vis_frames = []
vis_video_paths = []
vis_probabs = []
vis_true_labels = []
predictions = []
true_labels = []
for i in range(0,len(test_dataset)):
  sample_idx = i
  sample = test_dataset[sample_idx]

  # Extract frames from the selected video
  video_path = sample["video_path"]
  frames = test_dataset.extract_frames(video_path)  # Extract full frames for visualization
  first_frame = frames[0]  # Select the first frame from the list


  # Run model inference on the sample
  with torch.no_grad():
    inputs = sample["pixel_values"].unsqueeze(0).to(device)
    outputs = final_model(**{"pixel_values": inputs})
    predicted_label = torch.argmax(outputs.logits, dim=1).cpu().item()
    predictions.append(predicted_label)
    true_labels.append(sample["label"])
    true_label = categories[sample["label"]]
    probabilities = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]
    if i % 50 == 0:
        vis_probabs.append(probabilities)
        vis_true_labels.append(true_label)

  # Map predicted label to category
  predicted_class = categories[predicted_label]
  if i % 50 == 0:
      print(i)
      vis_predicted_classes.append(predicted_class)
      vis_frames.append(first_frame)
      vis_video_paths.append(video_path)



accuracy = accuracy_score(true_labels, predictions)
print(f"Accuracy: {accuracy}")
report = classification_report(true_labels, predictions, output_dict=True)
print(report)
if use_wandb:
    wandb.log({"test_accuracy": accuracy})
    wandb.log({"classification_report": report})
    wandb.finish()

# Visualization of selected videos
num_cols = 4  # Number of columns for the plot
num_rows = int(np.ceil(len(vis_frames) / num_cols))  # Calculate number of rows

fig, axes = plt.subplots(num_rows, num_cols, figsize=(10*num_cols, 10*num_rows))  # Adjust figsize

# Flatten the axes array for easier indexing
axes = axes.flatten()

for i, frame in enumerate(vis_frames):
    axes[i].imshow(frame, aspect='equal')
    title = f"Predicted: {vis_predicted_classes[i]}\nTrue:{vis_true_labels[i]}\nProbabs: {vis_probabs[i]}\n{os.path.basename(vis_video_paths[i])}"
    axes[i].set_title(title, fontsize=24)
    axes[i].axis("off")

plt.tight_layout()  # Adjust layout for better spacing
plt.subplots_adjust(wspace=0.5, hspace=0.5)
plt.show()
