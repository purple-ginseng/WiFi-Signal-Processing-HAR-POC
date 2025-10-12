#!/usr/bin/env python3
"""
Script to randomly select CSV files for training and validation datasets.
Selects 2 CSV files per activity from files matching pattern: wifisignal_data_{activity}_{timestamp}.csv
"""

import os
import random
import shutil
from collections import defaultdict
import re

def parse_filename(filename):
    """Extract activity and timestamp from filename."""
    pattern = r'wifisignal_data_(.+?)_(\d{8}_\d{6})\.csv'
    match = re.match(pattern, filename)
    if match:
        return match.group(1), match.group(2)
    return None, None

def main():
    # Get current directory
    data_dir = os.getcwd()
    
    # Create output directories
    train_dir = os.path.join(data_dir, 'small_train_data')
    valid_dir = os.path.join(data_dir, 'small_valid_data')
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(valid_dir, exist_ok=True)
    
    # Group files by activity
    activity_files = defaultdict(list)
    
    # Find all CSV files matching the pattern
    for filename in os.listdir(data_dir):
        if filename.startswith('wifisignal_data_') and filename.endswith('.csv'):
            activity, timestamp = parse_filename(filename)
            if activity and timestamp:
                activity_files[activity].append(filename)
    
    print(f"Found activities: {list(activity_files.keys())}")
    print(f"Files per activity: {[(activity, len(files)) for activity, files in activity_files.items()]}")
    
    # Select 3 files per activity
    selected_files = {'train': [], 'valid': []}
    
    for activity, files in activity_files.items():
        if len(files) < 3:
            print(f"Warning: Activity '{activity}' has only {len(files)} file(s), need at least 2")
            continue
            
        # Randomly select 3 files for this activity
        random.shuffle(files)
        selected = files[:3]
        
        # First file goes to train, second to validation
        selected_files['train'].append((activity, selected[0]))
        selected_files['valid'].append((activity, selected[1]))
        
        print(f"Activity '{activity}': train={selected[0]}, valid={selected[1]}")
    
    # Copy files to respective directories
    for split, file_list in selected_files.items():
        target_dir = train_dir if split == 'train' else valid_dir
        
        for activity, filename in file_list:
            src_path = os.path.join(data_dir, filename)
            dst_path = os.path.join(target_dir, filename)
            
            shutil.copy2(src_path, dst_path)
            print(f"Copied {filename} to {split} dataset")
    
    print(f"\nDataset creation complete!")
    print(f"Training files: {len(selected_files['train'])} (in {train_dir})")
    print(f"Validation files: {len(selected_files['valid'])} (in {valid_dir})")

if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)
    main()