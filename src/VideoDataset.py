import os
import torch
import numpy as np
from torch.utils.data import Dataset
import cv2

class VideoDataset(Dataset):
    def __init__(self, video_dir, feature_extractor, num_frames=8, resize=(224, 224), transform=None):
        """
        Args:
            video_dir (str): Path to dataset directory containing 'content/' and 'commercial/'.
            feature_extractor: AutoImageProcessor feature extractor for processing frames.
            num_frames (int): Number of frames to extract per video.
            resize (tuple): Resize frames to this resolution.
            transform: Additional PyTorch transforms (optional).
        """
        self.video_dir = video_dir
        self.feature_extractor = feature_extractor
        self.num_frames = num_frames
        self.resize = resize
        self.transform = transform

        # Load video file paths and labels
        self.video_files, self.labels = self._load_video_paths()

    def _load_video_paths(self):
        """Scan directories and create a list of (video_path, label) tuples."""
        video_files = []
        labels = []

        for label, category in enumerate(["content", "commercial"]):
            category_path = os.path.join(self.video_dir, category)
            for file in  sorted(os.listdir(category_path)): # Ensure consistent order
                if file.endswith(".mp4"):
                    video_files.append(os.path.join(category_path, file))
                    labels.append(label)  # content = 0, commercial = 1

        return video_files, labels

    def extract_frames(self, video_path):
        """Extract evenly spaced frames from the video file."""
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Handle short videos by duplicating last frame
        if total_frames < self.num_frames:
            frame_indices = np.linspace(0, total_frames - 1, total_frames, dtype=int).tolist()
            frame_indices += [total_frames - 1] * (self.num_frames - total_frames)  # Pad with last frame
        else:
            frame_indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)# Select evenly spaced frame indices

        frames = []
        for frame_index in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)  # Seek to the right frame
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
            frame = cv2.resize(frame, self.resize)  # Resize to model input size
            frames.append(frame)

        cap.release()
        return frames

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, idx):
        """Load video, extract frames, apply feature extractor, return tensor & label."""
        video_path = self.video_files[idx]
        label = self.labels[idx]

        frames = self.extract_frames(video_path)  # Extract frames
        inputs = self.feature_extractor(frames, return_tensors="pt")  # Preprocess
        pixel_values = inputs["pixel_values"].squeeze(0)  # Remove batch dim (1, T, C, H, W) -> (T, C, H, W)

        return {"pixel_values": pixel_values, "label": label, "video_path": video_path}
    

