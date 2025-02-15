# **Commercial vs. Content Classification**

This project classifies 10-second video clips from TV streams into "commercial" or "content" using a fine-tuned TimeSformer model.

## **Dataset Preparation**

1. **Collecting TV Streams:** Recorded 1-hour TV streams including timestamps from German channels (RTL, RTL2, VOX, N24, SUPER RTL, PROSIEBEN, SAT1, SPORT1, DMAX, SIXX, KABEL EINS, TELE 5, N24 DOKU, DISNEY CHANNEL, MTV, TLC, HGTV)
2. **Identifying Commercial Blocks:** Used AGF media plans to determine ad segments.
3. **Clipping Videos:** Used ffmpeg to cut recordings into 10-second clips.
4. **Categorizing Clips:**
- Labeled clips as commercial or content using a script.
- Manually verified the classification.
- Rejected ambiguous mixed clips.

5. **Dataset Splitting:** Used split_dataset.py to randomly split data into:
- Train: 70% (7440 content clips + 1902 commercial clips)
- Validation: 15% (1594 content clips + 408 commercial clips)
- Test: 15% (1595 content clips + 408 commercial clips)

`python split_dataset.py --input_dir=<input_dir> --output_dir=<output_dir> --train_ratio=0.7 --val_ratio=0.15 --test_ratio=0.15`

## **Model Training**

- **Backbone Model:** facebook/timesformer-base-finetuned-k400
- **Fine-tuning Method:** LoRA (reduces memory and compute requirements)
- **Feature Extraction:** AutoImageProcessor Feature Extractor (extracts and normalizes frames)
- **Optimization:** AdamW with weight decay
- **Evaluation Metrics:** F1-score, accuracy

### **Running Training**

`python train_commercial_content_classification.py --dataset_dir=<dataset_dir>  --models_dir=.<models_dir> --log_dir=<logs_dir> --learning_rate=2e-4 --epochs=10  --batch_size=4`

## **Classification & Evaluation**

`classify_commercial_content.py` can classify new videos or evaluate the model on the test dataset.

### **Running Classification**

`python scripts/classify_commercial_content.py --input_video path/to/video.mp4 --model_path models/best_model`

### **Evaluating on Test Data**

`python scripts/classify_commercial_content.py --test_data data/test --model_path models/best_model`

## **Installation & Setup**

**1. Clone the repository**

```
git clone https://github.com/HencAd/CV_project.git
cd CV_project
```

**2. Install dependencies**

`pip install torch torchvision decord transformers peft bitsandbytes wandb scikit-learn numpy opencv-python`

**2. Dataset**
Make sure your dataset is structured like this:
```
dataset/
│── train/
│   ├── commercial/
│   ├── content/
│── val/
│   ├── commercial/
│   ├── content/
│── test/
│   ├── commercial/
│   ├── content/
```

**3. Set up Weights & Biases (Optional)**

`wandb login`

## **Results & Visualizations**

- Training logs and metrics are available on W&B.
- Example classifications with model predictions.

## **Future Work**

- Improve classification accuracy with additional training data.
- Explore other video transformer models.
- Optimize inference speed for real-time classification.

## **License**

MIT License
