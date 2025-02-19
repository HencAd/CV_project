import os
import random
from torch.utils.data import Dataset, DataLoader
import torchvision.io as io
import numpy as np
import torch
from copy import deepcopy
from PIL import Image
from torchvision import transforms
from transformers import TimesformerForVideoClassification, Trainer, TrainingArguments
from decord import VideoReader, cpu

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel, PeftConfig

from tqdm import tqdm 
from sklearn.metrics import accuracy_score, classification_report

import wandb


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

wandb.login() 

config = {"lr": 0.0001, "batch_size": 4}
wandb.init(config=config) 

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

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
        
        return {"pixel_values": video_tensor, "labels": torch.tensor(label, dtype=torch.long)}



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

# Dataset paths
root_dir = "dataset"
train_dataset = AdDetectionDataset(root=root_dir, split="train")
val_dataset =  AdDetectionDataset(root=root_dir, split="val")
test_dataset =  AdDetectionDataset(root=root_dir, split="test")

batch_size = 4  
num_workers = 0  #Number of threads for data loading - set to 0 for Windows due to issues with multithreading caused by multiprocessing.

train_loader = DataLoader(
    train_dataset, 
    batch_size=batch_size, 
    shuffle=True, 
    num_workers=num_workers
)

val_loader= DataLoader(
    val_dataset,
    batch_size=batch_size,
    num_workers=num_workers
    )


test_loader= DataLoader(
    test_dataset,
    batch_size=batch_size,
    num_workers=num_workers
)


# Load pre-trained TimeSformer model
model = TimesformerForVideoClassification.from_pretrained(
        "facebook/timesformer-base-finetuned-k400", 
        ignore_mismatched_sizes=True,
        torch_dtype=torch.float16,
        num_labels=2
        )

# Prepare model for LoRA fine-tuning
base_model = prepare_model_for_kbit_training(model)
base_model.to(device)

lora_config = LoraConfig(
        lora_alpha=32,
        lora_dropout=0.1,
        r=16,
        bias="lora_only",
        target_modules="all-linear"
        )

p_model = get_peft_model(base_model, lora_config)
p_model.print_trainable_parameters()


#Creating a log folder for each training run
import datetime
current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
logging_dir = f"./logs/run_{current_time}"


#Saving additional plots
from sklearn.metrics import accuracy_score, f1_score, recall_score
def compute_metrics(eval_pred):
    """
    Computes and returns evaluation metrics based on model predictions.

    Args:
        eval_pred (tuple): A tuple containing two arrays:
            - logits (ndarray): Logits predicted by the model with shape (n_samples, n_classes).
            - labels (ndarray): True labels with shape (n_samples,).

    Returns:
        dict: A dictionary containing the computed metrics:
            - "eval_accuracy" (float): Model accuracy.
            - "eval_f1" (float): Weighted F1-score.
            - "eval_recall" (float): Weighted recall.

    """
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)

    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average="weighted")
    recall = recall_score(labels, predictions, average="weighted")
    
    
    wandb.log({"f1": f1, "recall": recall, "accuracy": accuracy})
    
    metrics = {
            "eval_accuracy": accuracy,
            "eval_f1": f1,
            "eval_recall": recall,
        }
    
    return metrics

wandb.watch(model, log="all", log_freq=100)

training_args = TrainingArguments(
    output_dir="./results", # Directory where training results and model checkpoints will be saved
    eval_strategy="steps", # Model evaluation is performed at the end of eval_steps
    save_strategy="steps", # The model is saved after each epoch
    eval_steps=500, # How often the model should be evaluated during training
    per_device_train_batch_size=4, # Batch size per device (GPU/CPU) during training
    per_device_eval_batch_size=4, # Batch size per device (GPU/CPU) during evaluation
    num_train_epochs=3, # Number of training epochs
    learning_rate=1e-4, # Initial learning rate for training
    gradient_accumulation_steps=4, # Number of steps before updating model weights (useful for large models)
    optim="paged_adamw_8bit", # AdamW optimizer in 8-bit mode for memory efficiency
    logging_dir=logging_dir, # Path to the directory where TensorBoard logs will be stored
    logging_steps=10, # Logging frequency in training steps
    weight_decay=0.01, # L2 regularization coefficient (prevents overfitting)
    report_to="wandb",  # Specifies where to report training metrics (TensorBoard)
    label_names=["labels"],  # Specifies the name of the label field in the dataset - use to compute matrix
    max_grad_norm=0.5,  # Maximum gradient norm for gradient clipping (prevents gradient explosion)
    warmup_ratio=0.03,  # Fraction of total steps allocated for learning rate warm-up
    lr_scheduler_type="cosine", # Scheduler that adjusts learning rate using a cosine function
    save_safetensors=True, # Saves the model in `safetensors` format for increased security
    save_total_limit=3, # Maximum number of checkpoints stored on disk (older ones are overwritten)
    fp16=True, # Enables 16-bit floating-point precision for memory efficiency and faster computation
    push_to_hub=False, # Determines whether the model should be automatically uploaded to Hugging Face Hub
    metric_for_best_model="eval_f1",  # Metric used to determine the best model
    greater_is_better=True,  # Indicates whether a higher metric value is better (True for F1-score)
    load_best_model_at_end=True,  # Loads the best model checkpoint at the end of training
    )

trainer = Trainer(
    model=p_model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
    )


#Train model
trainer.train()
trainer.evaluate()
wandb.finish()

p_model.config.id2label = {
    0: "content",
    1: "commercial"
    }

p_model.config.label2id = {v: k for k, v in p_model.config.id2label.items()}

full_model = p_model.merge_and_unload()

model_path = "./final_model"
full_model.save_pretrained(model_path)


final_model = TimesformerForVideoClassification.from_pretrained(model_path)

final_model.eval()
final_model.to(device)

#Classification on test set
predictions = []
true_labels = []

for batch in tqdm(test_loader, desc="Classification"):
    inputs = batch["pixel_values"].to(device) 
    labels = batch["labels"].squeeze().to(device)  

    with torch.no_grad():
        outputs = final_model(**{"pixel_values": inputs})
        predicted_labels = torch.argmax(outputs.logits, dim=1)

        predictions.extend(predicted_labels.cpu().numpy())
        true_labels.extend(labels.cpu().numpy())
        
accuracy = accuracy_score(true_labels, predictions)
print(f"Accuracy: {accuracy}")

report = classification_report(true_labels, predictions)
print(report)
