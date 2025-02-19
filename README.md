# Video Commercial/Content Detection with Fine-Tuned Model

## Table of Contents  
- [Project Overview](#1-project-overview)  
- [Dataset](#2-dataset)  
  - [Dataset Preparation](#dataset-preparation)  
  - [Dataset Splitting](#dataset-splitting)  
  - [Data Loader](#data-loader)  
- [Fine-Tuning](#3-fine-tuning)  
  - [Training Details](#training-details)  
  - [Model Evaluation](#model-evaluation)  
  - [Model Testing](#model-testing)  
  - [Results & Visualizations](#results--visualizations)  
- [Usage](#4-usage)  
- [Further Development](#5-further-development)
  
## 1. Project Overview

This project aims to fine-tune a pre-trained model for detecting whether a video segment is an advertisement ("commercial") or regular content ("content"). The model was trained using a custom dataset consisting of short video clips (10 seconds each), and the fine-tuned model can be used for detecting advertisements in new video inputs.

Additionally, a simple demo is provided, allowing users to upload videos and receive real-time predictions from the fine-tuned model (content vs commercial). This serves as a basic application to test the model's performance on unseen videos.

## 2. Dataset
### Dataset Preparation
1. **Collecting TV Streams:** Recorded 1-hour TV streams including timestamps from German channels (RTL, RTL2, VOX, N24, SUPER RTL, PROSIEBEN, SAT1, SPORT1, DMAX, SIXX, KABEL EINS, TELE 5, N24 DOKU, DISNEY CHANNEL, MTV, TLC, HGTV)
2. **Identifying Commercial Blocks:** Used AGF media plans to determine ad segments.
3. **Clipping Videos:** Used ffmpeg to cut recordings into 10-second clips.
4. **Categorizing Clips:**
- Labeled clips as commercial or content using a script.
- Manually verified the classification.
- Rejected ambiguous mixed clips.

### Dataset Splitting
- Train: 70% (7440 content clips + 1902 commercial clips)
- Validation: 15% (1594 content clips + 408 commercial clips)
- Test: 15% (1595 content clips + 408 commercial clips)

We used split_dataset.py script to randomly split data into train, validation and test splits.<br>
`python src/split_dataset.py --input_dir=<input_dir> --output_dir=<output_dir> --train_ratio=0.7 --val_ratio=0.15 --test_ratio=0.15`

### Data Loader
The videos from the dataset are loaded using the `decor` library, which samples 16 frames from each video and resizes them to 224x224 pixels for input into the model. The process ensures that each video is uniformly processed, maintaining the necessary consistency across the dataset.

## 3. Fine-Tuning

- **Backbone Model:** facebook/timesformer-base-finetuned-k400
- **Fine-tuning Method:** LoRA (reduces memory and compute requirements)
- **Feature Extraction:** We used custom processor for processing and normalizing the data, ensuring it replicates the behavior of the official processor.
- **Optimization:** AdamW with weight decay
- **Evaluation Metrics:** F1-score, accuracy
  <br><br><br>
The model was fine-tuned for 3 epochs with the following parameters:

- **Trainable Parameters**: 3,680,290
- **Total Parameters**: 124,811,554
- **Trainable Percentage**: 2.95%

### Training Details:

- **Epoch 1**:
    - Train Loss: 0.071
    - Eval Loss: 0.038
    - Avg. Gradient Norm: 0.33
    - Learning Rate: 1.3e-4
- **Epoch 2**:
    - Train Loss: 0.025
    - Eval Loss: 0.029
    - Avg. Gradient Norm: 0.08
    - Learning Rate: 6.9e-5
- **Epoch 3**:
    - Train Loss: 0.008
    - Eval Loss: 0.028
    - Avg. Gradient Norm: 0.002
    - Learning Rate: 5.7e-7

The fine-tuning progress was tracked using metrics such as the loss and gradient norm. By the end of training, the model showed promising results with a **validation accuracy of 98.7%**, **F1 score of 98.7%**, and **recall of 98.7%**.

### Model Evaluation:
- **Accuracy**: 98.7%
- **F1 Score**: 98.7%
- **Recall**: 98.7%
- **Loss**: 0.028

### Model Testing:
- **Accuracy**: 98.2%
- **F1 Score**: 98.2%
- **Recall**: 98.2%
<br><br>
The training, evaluation and test results indicate that the model is highly effective in distinguishing between advertisements and content, which makes it suitable for deployment in real-world scenarios.

#### Limitations & Challenges:
While the model performs well overall, there are specific cases where it struggles:  

1. **Social Awareness Commercials**: Some public service announcements (PSAs) and socially-driven ads are difficult to classify because they lack typical advertising cues such as logos, branding, or direct promotional messages.  
2. **Film-Like Ads**: Certain commercials resemble movie scenes or short films, making them visually and narratively similar to regular content. These ads do not include overlay text, product placements, or frequent logo appearances, which are often strong indicators of commercials.  
3. **Silent or Minimal-Text Advertisements**: Since our model primarily focuses on visual patterns, commercials that rely on subtle storytelling rather than explicit branding may be misclassified as content.  

These challenges indicate that the model primarily learns from visual patterns, but struggles with abstract or artistic commercial formats.  

### Results & Visualizations
<p float="left">
  <img src="https://github.com/HencAd/CV_project/blob/Ela/images/W%26B%20Chart%2019.02.2025%2C%2014_11_43.png" width="400"/>
  <img src="https://github.com/HencAd/CV_project/blob/Ela/images/W%26B%20Chart%2019.02.2025%2C%2014_11_52.png" width="400"/>
</p>
<img src="https://github.com/HencAd/CV_project/blob/Ela/images/W%26B%20Chart%2019.02.2025%2C%2014_12_05.png" width="400"/>
<br>
<img src="https://github.com/HencAd/CV_project/blob/Ela/images/examples1.png"/>
<img src="https://github.com/HencAd/CV_project/blob/Ela/images/W%26B%20Chart%2019.02.2025%2C%2014_11_06.png" width="400"/>

## 4. Usage

To run the model and interact with the demo, follow the instructions below.

### Installation:

```
    1. Clone the repository:
    git clone https://github.com/HencAd/CV_project.git
    cd CV_project

    2. Install requirements
    pip install -r requirements.txt

    3. Make sure your dataset is structured like this:
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

    3. Running Fine-tuning:
    python main.py

    4. Running the Demo:
    python app.py

    5. Running classification on test dataset, separately from main.py script,
       with visualization in matplotlib and logging example media to W&B

    python src/classify_test_dataset.py  --dataset_dir=<dataset_dir> --wandb_project=<WB_PROJECT> --wandb_run_name=<WB_RUN>

    6. Classify not labeled data from outside of dataset
    python src/classify_new_data.py --data_dir=<data_dir>
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

4. **Improving Detection of Film-Like Commercials**  
   - Introduce **temporal attention mechanisms** to better analyze how ads transition over time.  
   - Incorporate **scene detection algorithms** to identify cuts and pacing typical of commercials.  
   - Experiment with **contrastive learning** to distinguish film-like ads from actual movie scenes.
   - Expand the dataset by **adding more mislabeled or ambiguous commercials**, training the model to recognize edge cases better.  

   


