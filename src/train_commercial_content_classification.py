import os
import torch
import bitsandbytes as bnb
import wandb
import numpy as np
import argparse
import cv2
from pathlib import Path
from torchvision import transforms
from decord import VideoReader, cpu
from torch.utils.data import Dataset, DataLoader, Subset # Import Dataset and DataLoader
from transformers import AutoImageProcessor, TimesformerForVideoClassification, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from sklearn.metrics import accuracy_score, f1_score, recall_score
import VideoDataset
""" 
Requires GPU.

The script for fine-tuning facebook/timesformer-base-finetuned-k400 model to classify video clips into "commercial" or "content".

Timesformer-base-finetuned-k400 model is a transformer-based video model.
To enhance efficiency, we employ LoRA (Low-Rank Adaptation), which significantly reduces computational and memory overhead while maintaining model performance.

Our training pipeline is built using Hugging Face's Trainer (it takes care of dataset management, optimization, and evaluation)
The dataset consists of 10-second video clips categorized into "commercial" and "content," organized into train/, val/, and test/ sets.

To process video data, we extract evenly spaced frames and preprocess them using AutoImageProcessor Feature Extractor. 

The fine-tuning process optimizes the model using an adaptive learning strategy, saving the best-performing weights based on F1-score evaluation.

This approach ensures an efficient, scalable, and effective model for binary video classification, leveraging transformer-based architectures and parameter-efficient fine-tuning.

Example of use:
python train_commercial_content_classification.py --dataset_dir=<local_dir1>  --models_dir=<local_dir2> --log_dir=<local_dir3> --learning_rate=1e-5 --epochs=10 --batch_size=4 --wandb_project=<WANDB_PROJECT> --wandb_run_name=<WANDB_RUN_NAME> --logging_steps=50
"""

wandb.login()  # Authenticate with your W&B account

parser = argparse.ArgumentParser(description="Processing named arguments.")
parser.add_argument("--dataset_dir",   type=str,   required=True, help="Path to the dataset directory")
parser.add_argument("--model_dir",     type=str,   required=True, help="Path to the directory where the final model should be saved")
parser.add_argument("--log_dir",       type=str,   required=True, help="Path to the directory where the logs should be stored")
parser.add_argument("--learning_rate", type=float, default=2e-4,  help="learning rate")
parser.add_argument("--epochs",        type=int,   default=1,     help="how many epochs you would like to train")
parser.add_argument("--batch_size",    type=int,   default=4,     help="batch_size")
parser.add_argument("--wandb_project", type=str,                  help="if you want to log to W&B please provide the project name")
parser.add_argument("--wandb_run_name",type=str,                  help="if you want to log to W&B please provide the run name")
parser.add_argument("--logging_steps", type=int,   default=50,    help="how often the script should log during training")

args = parser.parse_args()

dataset_dir = args.dataset_dir
model_dir = args.model_dir
log_dir = args.log_dir
learning_rate = args.learning_rate
epochs = args.epochs
batch_size = args.batch_size
wandb_project = args.wandb_project
wandb_run_name = args.wandb_run_name
logging_steps = args.logging_steps

print(f"Used parameters: learning_rate: {learning_rate}, epochs: {epochs}, batch_size: {batch_size}.")

for phase in ['train','val', 'test']:
    phase_path = os.path.join(dataset_dir, phase)
    if not os.path.exists(phase_path):
        raise ("Dataset need to contain directories: 'train', 'val' and 'test'.")
    
    for category in ['commercial','content']:
        category_path = os.path.join(phase_path, category)
        if not os.path.exists(category_path):
            raise ("Each subfolder of dataset (train, val, test) needs to contain directories: 'commercial' and 'content'.")


use_wandb = 0
report_to = "none" #local disc (log_dir) on default
if (wandb_project and wandb_run_name):
    use_wandb = 1
    report_to = ["wandb"]

    wandb.init(project=wandb_project,
        name=wandb_run_name,
        config = {
            "learning_rate": learning_rate,
            "epochs": epochs,
            "batch_size": batch_size,
        })

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)

    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average="weighted")
    recall = recall_score(labels, predictions, average="weighted")

    if use_wandb:
        wandb.log({"f1": f1, "recall": recall, "accuracy": accuracy})

    metrics = {
        "eval_accuracy": accuracy,
        "eval_f1": f1,
        "eval_recall": recall,
    }

    print("Compute Metrics Output:", metrics)
    return metrics


# Create datasets
path_train=os.path.join(dataset_dir, 'train')
path_val=os.path.join(dataset_dir, 'val')

# Load the feature extractor
feature_extractor = AutoImageProcessor.from_pretrained("facebook/timesformer-base-finetuned-k400")

# Create dataset
train_dataset = VideoDataset.VideoDataset(video_dir=path_train, feature_extractor=feature_extractor)
val_dataset   = VideoDataset.VideoDataset(video_dir=path_val,   feature_extractor=feature_extractor)


base_model = TimesformerForVideoClassification.from_pretrained(
    "facebook/timesformer-base-finetuned-k400",
    num_labels=2,
    ignore_mismatched_sizes=True,
    torch_dtype=torch.float16,
)
base_model = prepare_model_for_kbit_training(base_model)  # Enable 8-bit

lora_config = LoraConfig(
    r=64, #how many parameters it will remember
    lora_dropout=0.1,
    bias="none",
    target_modules = 'all-linear'
)


model = get_peft_model(base_model, lora_config)
if torch.cuda.is_available():
    device = torch.device("cuda")
    model = model.to(device)
else:
    print("CUDA is not available. Using CPU instead.")
    device = torch.device("cpu")
    model = model.to(device)

if use_wandb:
    wandb.watch(model, log="all", log_freq=100)


training_args = TrainingArguments(
    output_dir=models_dir,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_steps=logging_steps,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=epochs,
    learning_rate=learning_rate,
    gradient_accumulation_steps=4,
    optim="paged_adamw_8bit",
    save_total_limit=2,
    fp16=True,
    push_to_hub=False,
    report_to=report_to,
    logging_dir=log_dir,
    metric_for_best_model="eval_f1",
    load_best_model_at_end=True,
    greater_is_better=True,
    label_names=["labels"]
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.evaluate()

if use_wandb:
    wandb.finish()

# Update the model's configuration with label names
model.config.id2label = {
    0: "content",
    1: "commercial"
}
model.config.label2id = {v: k for k, v in model.config.id2label.items()}
merged_model = model.merge_and_unload()
merged_model.save_pretrained(model_dir)

