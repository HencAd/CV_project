# Video Commercial/Content Detection with Fine-Tuned Model

## 1. Project Overview

This project aims to fine-tune a pre-trained model for detecting whether a video segment is an advertisement ("commercial") or regular content ("content"). The model was trained using a custom dataset consisting of short video clips (10 seconds each), and the fine-tuned model can be used for detecting advertisements in new video inputs.

Additionally, a simple demo is provided, allowing users to upload videos and receive real-time predictions from the fine-tuned model (content vs commercial). This serves as a basic application to test the model's performance on unseen videos.

## 2. Dataset

The dataset used consists of 10-second video clips divided into the following splits:
- **Train**: 1500 videos
- **Validation (val)**: 500 videos
- **Test**: 500 videos

These videos are loaded using the `decor` library, which samples 16 frames from each video and resizes them to 224x224 pixels for input into the model. The process ensures that each video is uniformly processed, maintaining the necessary consistency across the dataset.

## 3. Fine-Tuning

The model was fine-tuned for 3 epochs with the following parameters:

- **Trainable Parameters**: 3,680,290
- **Total Parameters**: 124,811,554
- **Trainable Percentage**: 2.95%

### Training Details:

- **Epoch 1**:
    - Loss: 0.7074
    - Gradient Norm: 1.43
    - Learning Rate: 3.77e-6
- **Epoch 2**:
    - Loss: 0.6694
    - Gradient Norm: 2.25
    - Learning Rate: 7.55e-6
- **Epoch 3**:
    - Loss: 0.6166
    - Gradient Norm: 1.25
    - Learning Rate: 1.13e-5

The fine-tuning progress was tracked using metrics such as the loss and gradient norm. By the end of training, the model showed promising results with a **validation accuracy of 95.35%**, **F1 score of 95.27%**, and **recall of 95.35%**.

### Model Evaluation:
- **Accuracy**: 95.35%
- **F1 Score**: 95.27%
- **Recall**: 95.35%
- **Loss**: 0.1209

The training results indicate that the model is highly effective in distinguishing between advertisements and content, which makes it suitable for deployment in real-world scenarios.

## 4. Usage

To run the model and interact with the demo, follow the instructions below.

### Installation:

```
    1. Clone the repository:
    git clone https://github.com/HencAd/CV_project.git
    cd CV_project
    2. Install requirements
    pip install -r requirements.txt

    3. Running Fine-tuning:
    python main.py

    4. Running the Demo:
    python app.py
```

## 5. Further Development

1. **Support for Larger Video Files and Audio Processing**
   - Allow users to upload longer videos, which will be automatically processed in parts.
   - Implement a mechanism to split video files into smaller segments, enabling easier and faster analysis.
   - Enable analysis of both video and audio in a single process, improving the accuracy of ad detection.

2. **Text Recognition in Ads (OCR)**
   - Implement OCR technology to identify brand names, products, and other text in ads.
   - Extend the video analysis capabilities by recognizing advertising text, which can help with more precise ad detection.

3. **Content Recommendation System**
   - Add a recommendation system that analyzes the video content and suggests similar movies or ads that might interest the user.
   - Use data from the video analysis to generate real-time recommendations, increasing user engagement and interactivity.