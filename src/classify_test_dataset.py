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
from decord import VideoReader, cpu


""" 
The script that classifies test dataset into 2 classes: "commercial" or "content" using the model trained by us https://huggingface.co/cvproject/final_model

It calculates accuracy, precision, recall, f1-score and prints classification_report.
It also collects every 50th video's picture + labels (true and predicted) and calculated probabilities and prints it out to visualize the result.

It can log to wandb if you pass wandb_project and wandb_run_name.

Example of use:
python classify_test_dataset.py --dataset_dir=<dataset_dir>
"""

wandb.login()  # Authenticate with your W&B account

parser1 = argparse.ArgumentParser(description="Processing named arguments.")
parser1.add_argument("--dataset_dir",   type=str,   required=True, help="Path to the dataset directory")
parser1.add_argument("--wandb_project", type=str,                  help="if you want to log to W&B please provide the project name")
parser1.add_argument("--wandb_run_name",type=str,                  help="if you want to log to W&B please provide the run name")

args1 = parser1.parse_args()

dataset_dir = args1.dataset_dir
wandb_project = args1.wandb_project
wandb_run_name = args1.wandb_run_name

print(f"Used parameters: dataset_dir: {dataset_dir}.")


class AdDetectionDataset(Dataset):
    """
    Dataset for ad detection in videos.

    Attributes:
        split_dir (str): Directory containing the dataset split (train, val, test).
        num_frames (int): Number of frames sampled from each video.
        video_paths (list[str]): List of video file paths.
        labels (list[int]): Corresponding labels for each video (0 for content, 1 for commercial).
        label_map (dict): Mapping from label names to numerical values.
    """
    def __init__(self, root: str, split: str, num_frames: int = 16):
        """
        Initialize dataset by loading video file paths and corresponding labels.

        :param root: Root directory containing dataset.
        :param split: Data split ('train', 'val', 'test').
        :param num_frames: Number of frames to sample from each video.
        """
        self.split_dir = os.path.join(root, split)
        self.num_frames = num_frames
        self.video_paths = []
        self.labels = []

        # Label mapping
        self.label_map = {'content': 0, 'commercial': 1 }
        
        # Traverse `content` and `commercial` directories
        for label_name, label in self.label_map.items():
            label_dir = os.path.join(self.split_dir, label_name)
            if not os.path.exists(label_dir):
                continue  
            
            for file in os.listdir(label_dir):
                video_path = os.path.join(label_dir, file)
                self.video_paths.append(video_path)
                self.labels.append(label)
        

    def __len__(self) -> int:
        """Returns the total number of videos in the dataset."""
        return len(self.video_paths)

    def __getitem__(self, idx: int) -> dict:
        """
        Loads a video, applies padding, and returns a tensor.

        :param idx: Index of the video in dataset.
        :returns: Dictionary containing video tensor and label.
        """
            
        video_path = self.video_paths[idx]
         
        label = self.labels[idx]
        video_tensor = load_video(video_path, self.num_frames)
        
        return {"pixel_values": video_tensor, "labels": torch.tensor(label, dtype=torch.long), "video_path": video_path}



def load_video(video_path: str, num_frames: int = 16) -> torch.Tensor:
    """
    Loads a video, selects frames, and applies transformations.
    
    :param video_path: Path to the video file.
    :param num_frames: Number of frames to sample.
    :return: A tensor of shape (num_frames, 3, 224, 224).
    """
    vr = VideoReader(video_path, ctx=cpu(0))  
    total_frames = len(vr)
    indices = torch.linspace(0, total_frames - 1, num_frames).long()  #We sample frames
    frames = [vr[int(i)].asnumpy() for i in indices]  #We go through the selected frames

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    #Applying transformations to the selected frames
    return torch.stack([transform(frame) for frame in frames])  # (num_frames, 3, 224, 224)


categories = ['content','commercial']

for phase in ['test']:
    phase_path = os.path.join(dataset_dir, phase)
    if not os.path.exists(phase_path):
        raise ("Dataset needs to contain directory 'test'.")
    
    for category in (categories):
        category_path = os.path.join(phase_path, category)
        if not os.path.exists(category_path):
            raise ("Each subfolder of dataset (test in this case) needs to contain directories: 'commercial' and 'content'.")


use_wandb = 0
if (wandb_project and wandb_run_name):
    use_wandb = 1

    wandb.init(project=wandb_project, name=wandb_run_name )


# Create dataset
test_dataset =  AdDetectionDataset(root=dataset_dir, split="test")

final_model = TimesformerForVideoClassification.from_pretrained('cvproject/final_model')

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
    frames = load_video(video_path, 8)  # Extract full frames for visualization
    first_frame = frames[0]  # Select the first frame from the list
    # ImageNet normalization parameters
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    # Denormalize: undo normalization
    first_frame = first_frame * std + mean  # Scale back to [0, 1]
    # Convert (C, H, W) → (H, W, C) for imshow
    first_frame = first_frame.permute(1, 2, 0).clamp(0, 1).numpy()


    # Run model inference on the sample
    with torch.no_grad():
        inputs = sample["pixel_values"].unsqueeze(0).to(device)
        outputs = final_model(**{"pixel_values": inputs})
        predicted_label = torch.argmax(outputs.logits, dim=1).cpu().item()
        predictions.append(predicted_label)
        true_labels.append(sample["labels"])
        true_label = final_model.config.id2label.get(sample["labels"], "Unknown")
        if true_label == 'Unknown':
            true_label = categories[sample["labels"]]
        probabilities = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]
        if i % 50 == 0:
            vis_probabs.append(probabilities)
            vis_true_labels.append(true_label)

    # Map predicted label to category
    predicted_class = final_model.config.id2label.get(predicted_label, "Unknown")
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

if use_wandb:

    wbdata = []

    for i, frame in enumerate(vis_frames):
        formatted_probs = "[" + ", ".join([f"{prob:.3f}" for prob in vis_probabs[i]]) + "]"
        title_wandb = f"Predicted:{vis_predicted_classes[i]}, True:{vis_true_labels[i]}\n Probabs: {formatted_probs}\n {os.path.basename(vis_video_paths[i])}"
        img = Image.fromarray(frame.astype('uint8')).convert('RGB')
        wimage = wandb.Image(frame, caption=title_wandb)
        wbdata.append(wimage)

    if len(wbdata) > 0:
        wandb.init(project=wandb_project, name=(wandb_run_name + '_media') )
        wandb.log({"examples": wbdata})
        wandb.finish()
