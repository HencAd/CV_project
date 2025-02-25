
from flask import Flask, render_template, request, jsonify
import os
import torch
from torchvision import transforms
from peft import PeftModel, PeftConfig
from transformers import TimesformerForVideoClassification, AutoImageProcessor
from decord import VideoReader, cpu
import torch.nn.functional as F
import cv2
import numpy as np
from PIL import Image
import imageio
from einops import rearrange, repeat, einsum


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


def init_model():
    # Load the pre-trained model for video classification with attention outputs
    model = TimesformerForVideoClassification.from_pretrained('cvproject/final_model', output_attentions=True)
    
    return model.eval()

def overlay_attention(image, attention_map):
    """
    Overlays a normalized attention map on a given image using a colormap for better visualization.
    
    Parameters:
        image (PIL.Image or np.ndarray): The image on which the attention map will be overlayed. 
                                         It can be a PIL image or a NumPy array.
        attention_map (np.ndarray): A 1D or 2D NumPy array representing the attention map. 
                                    It should either be flat or a 2D array of the same size as the image's resolution.
    
    Returns:
        PIL.Image: The resulting image with the attention map overlayed, in PIL format.

    Raises:
        ValueError: If the attention map is None or empty.
    
    Notes:
        - The attention map is resized to match the dimensions of the input image.
        - The colormap 'JET' is used for better visualization of the attention map.
        - The final image is a blend of the original image and the attention map, with the attention map contributing 40% and the original image contributing 60%.
    """
    if isinstance(image, np.ndarray):
        height, width = image.shape[:2]  # NumPy
    else:
        width, height = image.size  # PIL
    
    # Validate the attention map
    if attention_map is None or attention_map.size == 0:
        raise ValueError("attention_map is null!")
    
    attention_map = attention_map.astype(np.float32) # Convert attention map to float32 for precision
    
    # If the attention map is 1D, reshape it to 2D (square) based on its length
    if len(attention_map.shape) == 1:
        side = int(np.sqrt(attention_map.shape[0]))
        attention_map = attention_map[:side**2].reshape(side, side)

    # Resize the attention map to the dimensions of the image
    attention_map = cv2.resize(attention_map, (width, height))
    # Normalize the attention map to [0, 1]
    attention_map = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min())
   
    # Convert PIL image to NumPy
    image_np = np.array(image).astype(np.float32) / 255.0
    
    # Apply colormap
    attention_colormap = cv2.applyColorMap((attention_map * 255).astype(np.uint8), cv2.COLORMAP_JET)
    attention_colormap = attention_colormap.astype(np.float32) / 255.0
    
    # Blend images
    blended_image = 0.6 * image_np + 0.4 * attention_colormap
    blended_image = (blended_image * 255).astype(np.uint8)
    
    return Image.fromarray(blended_image)


def combine_divided_attention(attn_t, attn_s):
    """
    Combines temporal and spatial attention matrices by averaging over heads, adding residual connections, 
    and normalizing the attention matrices. Then, it merges the spatial and temporal attentions and extracts 
    the CLS (first token) attention, averaging over frames.

    Args:
    attn_t (torch.Tensor): Temporal attention matrix (batch, heads, tokens, tokens).
    attn_s (torch.Tensor): Spatial attention matrix (batch, heads, tokens, tokens).

    Returns:
    torch.Tensor: Combined attention matrix (batch, tokens, tokens, tokens).
    """
    # Temporal attention: average over heads
    attn_t = attn_t.mean(dim=1)  # (batch, tokens, tokens)
    # Add CLS token as identity matrix (because it refers to itself)
    I = torch.eye(attn_t.size(-1)).unsqueeze(0)
    attn_t = torch.cat([I, attn_t], 0)
    # Add residual connection
    attn_t = attn_t + torch.eye(attn_t.size(-1))[None, ...]
    # Renormalization
    attn_t = attn_t / attn_t.sum(-1, keepdim=True)

    # Spatial attention: average over heads
    attn_s = attn_s.mean(dim=1)
    # Add residual connection and renormalize
    attn_s = attn_s + torch.eye(attn_s.size(-1))[None, ...]
    attn_s = attn_s / attn_s.sum(-1, keepdim=True)

    # Combine spatial and temporal attention
    attn_ts = torch.einsum('tpk, ktq -> ptkq', attn_s, attn_t)

    # Extract CLS attention (first token) and average over frames
    attn_cls = attn_ts[0, :, :, :]  # (tokens, tokens)

    attn_cls_a = attn_cls.mean(dim=0)  # (tokens,)

    # Change dimensions order
    attn_cls_a = attn_cls_a.permute(1, 0)  # (8, 393)

    # Add dimensions and expand
    attn_cls_a = attn_cls_a.unsqueeze(0).unsqueeze(-1)  # (1, 8, 393, 1)
    attn_cls_a = attn_cls_a.expand(-1, -1, -1, 8)

    # Add CLS attention back
    attn_ts = torch.cat([attn_cls_a, attn_ts[1:, :, :, :]], dim=0)

    return attn_ts


def process_video(video_path, num_frames=16):
    """
    Processes a video to generate attention maps based on temporal and spatial attention,
    creates a heatmap by overlaying the attention maps on the frames, and generates a GIF of the heatmap.

    Args:
    video_path (str): The path to the input video.
    num_frames (int): The number of frames to extract from the video (default is 16).

    Returns:
    str: The predicted label for the video.
    float: The confidence of the predicted label.
    """
    model = init_model()
    model.to("cuda")
    model.half()  # Convert model to float16 for faster computation

    # Prepare video using VideoReader 
    vr = VideoReader(video_path, ctx=cpu(0)) 
    total_frames = len(vr)
    indices = torch.linspace(0, total_frames - 1, num_frames).long()
    frames = [vr[int(i)].asnumpy() for i in indices]

    # Image transformation
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])

    # Create tensor from frames
    video_tensor = torch.stack([transform(frame) for frame in frames])  # (num_frames, 3, 224, 224)
    video_tensor = video_tensor.unsqueeze(0)  # (1, num_frames, 3, 224, 224)
    video_tensor = video_tensor.to("cuda").half()

    # -----------------------------
    # Registering hooks for attention
    # -----------------------------
    time_attentions = []
    space_attentions = []
    hooks = []

    def get_attn_t(module, input, output):
        time_attentions.append(output.detach().cpu())

    def get_attn_s(module, input, output):
        space_attentions.append(output.detach().cpu())

    # Register hooks for temporal and spatial attention
    for name, module in model.named_modules():
        if 'temporal_attention.attention.attn_drop' in name:
            hooks.append(module.register_forward_hook(get_attn_t))
        elif 'attention.attention.attn_drop' in name:
            hooks.append(module.register_forward_hook(get_attn_s))
    #print(hooks, 'hooks')
    # -----------------------------
    # Forward pass through the model
    # -----------------------------
    with torch.no_grad():
        outputs = model(pixel_values=video_tensor, output_attentions=True)

        # Remove hooks after forward pass
        for h in hooks:
            h.remove()

        # Classification
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)
        confidence, predicted_class = torch.max(probs, dim=1)
        print(f"Predicted probabilities: {probs}")
        print(f"Predicted class: {predicted_class}, Confidence: {confidence.item() * 100:.2f}%")
        label = model.config.id2label[predicted_class.item()]

    # -----------------------------
    # Combining spatial and temporal attention
    # -----------------------------
    combined_attentions = []
    # Assume the number of temporal and spatial attention maps is the same
    for attn_t, attn_s in zip(time_attentions, space_attentions):
        combined_attentions.append(combine_divided_attention(attn_t, attn_s))

    # Multiply attention maps step-by-step
    # Set dimensions (p - number of tokens, t - number of frames)
    p, t = combined_attentions[0].shape[0], combined_attentions[0].shape[1]
    result = torch.eye(p * t)
    for attention in combined_attentions:
        # Reshape attention matrix to shape ((p*t), (p*t))
        attention = rearrange(attention, 'p1 t1 p2 t2 -> (p1 t1) (p2 t2)')
        result = torch.matmul(attention, result)
    
    # Przywracamy kształt: (p, t, p, t)
    mask = rearrange(result, '(p1 t1) (p2 t2) -> p1 t1 p2 t2', p1=p, p2=p)
   
    # Average over the frame dimension
    mask = mask.mean(dim=1)
    
    # Skip the first token (CLS)
    mask = mask[0, 1:, :]
    
    mask = mask.mean(dim=1) 
    width = int(mask.size(0)**0.5)
    
    num_patches = mask.size(0)
    num_patches_h = int(mask.size(0)**0.5)
    num_patches_w = num_patches_h
    
    if num_patches > num_patches_h * num_patches_w:
        mask = mask[:num_patches_h * num_patches_w]
    
    # Reshape to (H, W)
    mask = mask.reshape(num_patches_h, num_patches_w)

    mask = mask.detach().cpu().numpy() # Convert PyTorch tensor to NumPy array
    mask = mask / np.max(mask)

    # -----------------------------
    #  GENERATE GIF – Overlay Attention Map on Frames
    # -----------------------------
    
    # Prepare frames with overlaid attention map
    heatmap_frames = []
    mask_frames = [mask[..., i] for i in range(mask.shape[-1])]
    for img, mask_frame in zip(frames, mask_frames):
        attn_img = overlay_attention(img, mask_frame)
        heatmap_frames.append(np.array(attn_img))

    # Save GIF with attention heatmap frames
    imageio.mimsave("static/attention_heatmap.gif", heatmap_frames, duration=8, loop=0)

    return label, confidence.item()

























if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
