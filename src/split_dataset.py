import os
import shutil
import random
from pathlib import Path
import argparse

""" Script breaks down "commercial-content" categories dataset into three distinct dataset splits: test, train and val.
It assumes that input directory contains 2 directories: 'commercial' and 'content' (one for each category)
Script takes as arguments:
--input_dir - a path to the dataset directory to split
--output_dir - a path to the directory where splitted dataset should be saved
--train_ratio - a train part ratio, number between 0 and 1, default is 0.7
--val_ratio - a validation part ratio, number between 0 and 1, default is 0.15
--test_ratio - a test part ratio, number between 0 and 1, default is 0.15
"""

parser = argparse.ArgumentParser(description="Processing named arguments.")
parser.add_argument("--input_dir", type=str, required=True, help="Path to the dataset directory you want to split")
parser.add_argument("--output_dir", type=str, required=True, help="Path to the directory where you want to save splitted dataset")
parser.add_argument("--train_ratio", type=float, default=0.7, help="train part ratio, number between 0 and 1, default is 0.7 (a part-to-whole dataset)")
parser.add_argument("--val_ratio", type=float, default=0.15, help="validation part ratio, number between 0 and 1, default is 0.15 (a part-to-whole dataset)")
parser.add_argument("--test_ratio", type=float, default=0.15, help="test part ratio, number between 0 and 1, default is 0.15 (a part-to-whole dataset)")
args = parser.parse_args()

def split_dataset(input_dir, output_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    if not (0 <= train_ratio <= 1 and 0 <= val_ratio <= 1 and 0 <= test_ratio <= 1):
        raise ValueError("Ratios must be between 0 and 1.")
    if train_ratio + val_ratio + test_ratio != 1:
        raise ValueError("Ratios must sum up to 1.")
    print(f"Used parameters: train_ratio: {train_ratio}, test_ratio: {test_ratio}, validation_ratio: {val_ratio}.")
      
    for category in ['commercial','content']:
        category_path = os.path.join(input_dir, category)
        if not os.path.exists(category_path):
            print(f"Warning: {category_path} does not exist.")
            continue
        
        files = list(Path(category_path).glob("*"))
        print(f"Found {len(files)} files for category {category} to split.")
        random.shuffle(files)
        
        train_split = int(len(files) * train_ratio)
        val_split = int(len(files) * (train_ratio + val_ratio))
        
        splits = {
            "train": files[:train_split],
            "val": files[train_split:val_split],
            "test": files[val_split:]
        }
        
        for split_name, split_files in splits.items():
            split_path = Path(os.path.join(output_dir,split_name,category))
            split_path.mkdir(parents=True, exist_ok=True)
            
            for file in split_files:
                print(f"Processing {split_path} / {file.name}.")
                shutil.copy(file, split_path / file.name)
                
    print("Dataset split complete.")

split_dataset(args.input_dir, args.output_dir, args.train_ratio, args.val_ratio, args.test_ratio)