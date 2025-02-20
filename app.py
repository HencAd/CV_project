from flask import Flask, render_template, request, jsonify
import os
import torch
from torchvision import transforms
from peft import PeftModel, PeftConfig
from transformers import TimesformerForVideoClassification, AutoImageProcessor
from decord import VideoReader, cpu
import torch.nn.functional as F

app = Flask(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Folder, w którym będą przechowywane przesyłane pliki
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

#Home page
@app.route('/')
def index():
    return render_template('index.html')

# Endpoint for upload files
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # save file in directory
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
    processor = AutoImageProcessor.from_pretrained("facebook/timesformer-base-finetuned-k400")
    vr = VideoReader(video_path, ctx=cpu(0))  
    total_frames = len(vr)
    indices = torch.linspace(0, total_frames - 1, num_frames).long()
    frames = [vr[int(i)].asnumpy() for i in indices]
    
    #inputs = processor(frames, return_tensors="pt").to(device)  # Preprocess
    #pixel_values = inputs["pixel_values"].to(device).half()
    
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    
    video_tensor = torch.stack([transform(frame) for frame in frames])  # (num_frames, 3, 224, 224)
    video_tensor = video_tensor.unsqueeze(0)  # (1, num_frames, 3, 224, 224)
    video_tensor = video_tensor.to("cuda").half()
    #print(f"Video tensor shape: {video_tensor.shape}")
    #print(f"Tensor dtype: {video_tensor.dtype}")
    #print(f"Device video: {video_tensor.device}")
    #print(f"Device model: {model.device}")
    with torch.no_grad():
        outputs = model(**{"pixel_values": video_tensor})
        #outputs = model(**{"pixel_values": pixel_values})
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)  # Softmax na wynikach
        confidence, predicted_class = torch.max(probs, dim=1)  # Pobranie wartości i indeksu

        print(f"Predicted probabilities: {probs}")
        print(f"Predicted class: {predicted_class}, Confidence: {confidence.item() * 100:.2f}%")
    
        label =  model.config.id2label[predicted_class.item()]
    
    return label, confidence.item()


def init_model():
    #model_path="./final_model"
    model = TimesformerForVideoClassification.from_pretrained('cvproject/final_model')
    
    return model.eval()

    
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
