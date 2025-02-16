from flask import Flask, render_template, request, jsonify
import os
import torch
from torchvision import transforms
from peft import PeftModel, PeftConfig
from transformers import TimesformerForVideoClassification
from decord import VideoReader, cpu
import torch.nn.functional as F

app = Flask(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Folder, w którym będą przechowywane przesyłane pliki
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Strona główna
@app.route('/')
def index():
    return render_template('index.html')

# Endpoint do przesyłania plików
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Zapisz plik w wybranym folderze
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(video_path)

    result, confidence  = process_video(video_path)
    confidence_percentage = round(confidence * 100, 1) 

    os.remove(video_path)

    return jsonify({'result': result, 'confidence': confidence_percentage})

def process_video(video_path, num_frames=16):

    model = init_model()
    model.to(device)
    model.to(torch.float16)
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
    video_tensor = torch.stack([transform(frame) for frame in frames])  # (num_frames, 3, 224, 224)
    video_tensor = video_tensor.unsqueeze(0)  # (1, num_frames, 3, 224, 224)
    video_tensor = video_tensor.to("cuda").half()
    print(f"Video tensor shape: {video_tensor.shape}")
    print(f"Tensor dtype: {video_tensor.dtype}")
    print(f"Device video: {video_tensor.device}")
    print(f"Device model: {model.device}")
    with torch.no_grad():
        logits = model(video_tensor).logits
        probs = F.softmax(logits, dim=1)
        confidence, predicted_class = torch.max(probs, dim=1)

    label = "content" if predicted_class.item() == 0 else "commercial"
    return label, confidence.item()


def init_model():

    base_model = TimesformerForVideoClassification.from_pretrained(
        "facebook/timesformer-base-finetuned-k400", 
        device_map="auto",
        ignore_mismatched_sizes=True,
        torch_dtype=torch.float16
        )
    
    base_model.classifier = torch.nn.Linear(base_model.config.hidden_size, 2)
    base_model.num_labels = 2
    base_model.config.pretraining_tp=1

    model_path="./final_model"

    peft_config = PeftConfig.from_pretrained(model_path) 
    final_model = PeftModel.from_pretrained(base_model,model_path, config=peft_config) 
    
    return final_model.eval()

    
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
