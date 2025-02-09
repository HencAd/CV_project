import os
from torch.utils.data import Dataset, DataLoader
import torchvision.io as io
import numpy as np
import torch

from PIL import Image
from torchvision import transforms
from transformers import TimesformerForVideoClassification, BitsAndBytesConfig, Trainer, TrainingArguments
from decord import VideoReader, cpu

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel, PeftConfig



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class AdDetectionDataset(Dataset):
    def __init__(self, root: str, split:str, num_frames=16):
        """
        Inicjalizuje dataset, ładuje ścieżki do plików wideo i przypisuje etykiety.

        :param root: Główny katalog, w którym znajdują się dane (np. "dataset").
        :param split: Określa, którą część danych ładować: 'train', 'validate', 'test'.
        :param num_frames: Liczba klatek do próbkowania.
        """
        self.split_dir = os.path.join(root, split)
        self.num_frames = num_frames
        self.video_paths = []
        self.labels = []

        #Mapowanie etykiet
        self.label_map = {'content': 0, 'commercial': 1 }
        
        # Przeszukiwanie folderów `content` i `commercial`
        for label_name, label in self.label_map.items():
            label_dir = os.path.join(self.split_dir, label_name)
            if not os.path.exists(label_dir):
                continue  # Jeśli folder nie istnieje, pomijamy
            
            for file in os.listdir(label_dir):
                video_path = os.path.join(label_dir, file)
                self.video_paths.append(video_path)
                self.labels.append(label)
        

    def __len__(self):
        """Zwraca liczbę elementów w zbiorze danych (liczba filmów)."""
        return len(self.video_paths)

    def __getitem__(self, idx):
        """
        Wczytuje wideo, stosuje padding i zwraca tensor.

        :param idx: Indeks pliku w zbiorze danych.
        :returns: Para (wideo, etykieta).
        """
            
        video_path = self.video_paths[idx]
         
        label = self.labels[idx]
        video_tensor = load_video(video_path, self.num_frames)
        
        #print(video_tensor.shape)
        return {"pixel_values": video_tensor, "labels": torch.tensor(label, dtype=torch.long)}



def load_video(video_path, num_frames=16):
    
    vr = VideoReader(video_path, ctx=cpu(0))  
    total_frames = len(vr)
    indices = torch.linspace(0, total_frames - 1, num_frames).long()  # Losujemy klatki
    frames = [vr[int(i)].asnumpy() for i in indices]  # Przechodzimy przez wybrane klatki

    # Zastosowanie transformacji
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor()
        ])

    # Stosowanie transformacji na wybranych klatkach
    return torch.stack([transform(frame) for frame in frames])  # (num_frames, 3, 224, 224)
    
root_dir = "dataset"

# Tworzenie instancji datasetu dla treningu
train_dataset = AdDetectionDataset(root=root_dir, split="train")
val_dataset =  AdDetectionDataset(root=root_dir, split="val")
test_dataset =  AdDetectionDataset(root=root_dir, split="test")

#Sprawdzenie
#print(len(train_dataset))
#print(len(test_dataset ))
#video1, label1= train_dataset[0]
#print(video1, label1, '1element')



batch_size = 4  
num_workers = 0  # Liczba wątków do wczytywania danych - dla Windows 0 bo jest problem z wielowatkowscia wynikajaca z działania multiprocessingu

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

for idx in range(3):  # Przykład dla pierwszych 3 elementów
    example = train_dataset[idx]
    print(f"Example {idx}:")
    print(f"Pixel Values: {example['pixel_values'].shape}")
    print(f"Labels: {example['labels']}")
print(f"Train Dataset Size: {len(train_dataset)}")


bnb_config = BitsAndBytesConfig(
     load_in_4bit=True,
     bnb_4bit_quant_type="nf4",
     bnb_4bit_compute_dtype=torch.bfloat16  
)


model = TimesformerForVideoClassification.from_pretrained(
        "facebook/timesformer-base-finetuned-k400", 
        device_map="auto",
        ignore_mismatched_sizes=True,
        torch_dtype=torch.float16
        )
model.classifier = torch.nn.Linear(model.config.hidden_size, 2)
model.num_labels = 2

print(model)

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

training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=1,
    learning_rate=2e-5,
    gradient_accumulation_steps=4,
    optim="paged_adamw_8bit",
    logging_dir="./logs",
    logging_steps=10,
    weight_decay=0.001,
    report_to="tensorboard",
    max_grad_norm=0.5,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    save_safetensors=True,
    save_total_limit=2,
    fp16=True,
    push_to_hub=False
    )

trainer = Trainer(
    model=p_model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
    )


#Gdy model wytrenowany to zakomentowac
trainer.train()

model_path="./final_model"
trainer.save_model(model_path)
print('after')
######


from tqdm import tqdm 
from sklearn.metrics import accuracy_score, classification_report
peft_config = PeftConfig.from_pretrained(model_path) 

#Połącz model z adapterem
final_model = PeftModel.from_pretrained(model,model_path, config=peft_config) 

final_model.eval()


predictions = []
true_labels = []

# Iteracja po zbiorze testowym
for batch in tqdm(test_loader, desc="Classification"):
    inputs = batch["pixel_values"].to(device)
    labels = batch["labels"].to(device)

    # Wykonaj predykcję
    with torch.no_grad():
        outputs = final_model(**{"pixel_values": inputs})
        predicted_labels = torch.argmax(outputs.logits, dim=1)

        # Zbierz predykcje i prawdziwe etykiety
        predictions.extend(predicted_labels.cpu().numpy())
        true_labels.extend(labels.cpu().numpy())

# Oblicz dokładność
accuracy = accuracy_score(true_labels, predictions)
print(f"Dokładność: {accuracy}")

# Wygeneruj raport klasyfikacji
report = classification_report(true_labels, predictions)
print(report)
