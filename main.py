import os
from torch.utils.data import Dataset, DataLoader
import torchvision.io as io
import numpy as np
import torch
from copy import deepcopy
from PIL import Image
from torchvision import transforms
from transformers import TimesformerForVideoClassification, BitsAndBytesConfig, Trainer, TrainingArguments
from decord import VideoReader, cpu

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel, PeftConfig

from tqdm import tqdm 
from sklearn.metrics import accuracy_score, classification_report

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


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
                continue  # Jeśli folder nie istnieje, pomijamy
            
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
        
        #print(video_tensor.shape)
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
        transforms.ToTensor()
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



# bnb_config = BitsAndBytesConfig(
#      load_in_4bit=True,
#      bnb_4bit_quant_type="nf4",
#      bnb_4bit_compute_dtype=torch.bfloat16  
# )

# Load pre-trained TimeSformer model
model = TimesformerForVideoClassification.from_pretrained(
        "facebook/timesformer-base-finetuned-k400", 
        device_map="auto",
        ignore_mismatched_sizes=True,
        torch_dtype=torch.float16
        )
# Adjust model for binary classification
model.classifier = torch.nn.Linear(model.config.hidden_size, 2)
model.num_labels = 2

# print(model)

# Prepare model for LoRA fine-tuning
model.config.pretraining_tp=1
base_model = prepare_model_for_kbit_training(model)

peft_config = LoraConfig(
        lora_alpha=32,
        lora_dropout=0.05,
        r=16,
        bias="lora_only",
        target_modules="all-linear"
        )

p_model = get_peft_model(base_model, peft_config)
p_model.print_trainable_parameters()


#Creating a log folder for each training run
import datetime
current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
logging_dir = f"./logs/run_{current_time}"

#Printing evaluation metrics during training in the terminal
from transformers import TrainerCallback
class CustomCallback(TrainerCallback):
        
    def __init__(self, trainer) -> None:
        super().__init__()
        self._trainer = trainer
                                
    def on_epoch_end(self, args, state, control, **kwargs):
        if control.should_evaluate:
            control_copy = deepcopy(control)
            self._trainer.evaluate(eval_dataset=self._trainer.train_dataset, metric_key_prefix="train")
            return control_copy


#Saving additional plots
from sklearn.metrics import accuracy_score, f1_score, recall_score
def compute_metrics(eval_pred):
    """Oblicza i zwraca metryki dla ewaluacji."""
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)

    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average="weighted")
    recall = recall_score(labels, predictions, average="weighted")

    metrics = {
            "eval_accuracy": accuracy,
            "eval_f1": f1,
            "eval_recall": recall,
        }
    
    return metrics


training_args = TrainingArguments(
    output_dir="./results", # Directory where training results and model checkpoints will be saved
    eval_strategy="steps", # Model evaluation is performed at the end of eval_steps
    save_strategy="epoch", # The model is saved after each epoch
    eval_steps=500, # How often the model should be evaluated during training
    per_device_train_batch_size=4, # Batch size per device (GPU/CPU) during training
    per_device_eval_batch_size=4, # Batch size per device (GPU/CPU) during evaluation
    num_train_epochs=3, # Number of training epochs
    learning_rate=2e-5, # Initial learning rate for training
    gradient_accumulation_steps=4, # Number of steps before updating model weights (useful for large models)
    optim="paged_adamw_8bit", # AdamW optimizer in 8-bit mode for memory efficiency
    logging_dir=logging_dir, # Path to the directory where TensorBoard logs will be stored
    logging_steps=10, # Logging frequency in training steps
    weight_decay=0.001, # L2 regularization coefficient (prevents overfitting)
    report_to="tensorboard",  # Specifies where to report training metrics (TensorBoard)
    label_names=["labels"],  # Specifies the name of the label field in the dataset - use to compute matrix
    max_grad_norm=0.5,  # Maximum gradient norm for gradient clipping (prevents gradient explosion)
    warmup_ratio=0.03,  # Fraction of total steps allocated for learning rate warm-up
    lr_scheduler_type="cosine", # Scheduler that adjusts learning rate using a cosine function
    save_safetensors=True, # Saves the model in `safetensors` format for increased security
    save_total_limit=2, # Maximum number of checkpoints stored on disk (older ones are overwritten)
    fp16=True, # Enables 16-bit floating-point precision for memory efficiency and faster computation
    push_to_hub=False # Determines whether the model should be automatically uploaded to Hugging Face Hub
    )

trainer = Trainer(
    model=p_model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
    )

trainer.add_callback(CustomCallback(trainer)) 

#Train model
trainer.train()


# Save final model
model_path="./final_model"
trainer.save_model(model_path)

######

# Load fine-tuned model for evaluation
peft_config = PeftConfig.from_pretrained(model_path) 
final_model = PeftModel.from_pretrained(model,model_path, config=peft_config) 
final_model.eval()

# Evaluate on test set
predictions = []
true_labels = []

for batch in tqdm(test_loader, desc="Classification"):
    inputs = batch["pixel_values"].to(device)
    labels = batch["labels"].to(device)

    with torch.no_grad():
        outputs = final_model(**{"pixel_values": inputs})
        predicted_labels = torch.argmax(outputs.logits, dim=1)

        predictions.extend(predicted_labels.cpu().numpy())
        true_labels.extend(labels.cpu().numpy())

accuracy = accuracy_score(true_labels, predictions)
print(f"Accuracy: {accuracy}")

report = classification_report(true_labels, predictions)
print(report)
