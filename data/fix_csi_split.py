#!/usr/bin/env python3
"""
Fix CSI train/validation split to ensure both splits have samples from all sessions.

Current problem:
- Training: July 27 + July 28 sessions (mixed)
- Validation: Only July 28 session
- Result: Model can't generalize (40% train vs 6.6% val accuracy)

Solution:
- Both splits should have samples from BOTH sessions
- Split each session's data separately, then combine
"""

import shutil
from pathlib import Path
import random

# Paths
TRAIN_DIR = Path("small_train_data_csi")
VALID_DIR = Path("small_valid_data_csi")
NEW_TRAIN_DIR = Path("small_train_data_csi_fixed")
NEW_VALID_DIR = Path("small_valid_data_csi_fixed")

# Create new directories
NEW_TRAIN_DIR.mkdir(exist_ok=True)
NEW_VALID_DIR.mkdir(exist_ok=True)

print("=== Fixing CSI Train/Validation Split ===\n")

# Get all files from both directories
train_files = list(TRAIN_DIR.glob("*.csv"))
valid_files = list(VALID_DIR.glob("*.csv"))

all_files = train_files + valid_files

print(f"Total files: {len(all_files)}")
print(f"  - From training: {len(train_files)}")
print(f"  - From validation: {len(valid_files)}")

# Group by activity and session
activity_session_files = {}
for file in all_files:
    # Extract activity and session from filename
    # Format: esp32_csi_{activity}_{timestamp}.csv
    parts = file.stem.split('_')
    activity = parts[2]  # falling, sitting, etc.
    timestamp = parts[3]  # 20250727HHMMSS
    session = timestamp[:8]  # 20250727

    key = (activity, session)
    if key not in activity_session_files:
        activity_session_files[key] = []
    activity_session_files[key].append(file)

print(f"\n📊 Files by Activity and Session:")
for (activity, session), files in sorted(activity_session_files.items()):
    print(f"  {activity:10s} | Session {session}: {len(files)} files")

# Strategy: For each (activity, session) pair, split 80/20 into train/valid
random.seed(42)
train_count = 0
valid_count = 0

print(f"\n🔄 Creating Balanced Split (80% train, 20% validation):")

for (activity, session), files in activity_session_files.items():
    # Shuffle files for random split
    files_shuffled = files.copy()
    random.shuffle(files_shuffled)

    # Split 80/20
    n_train = max(1, int(len(files) * 0.8))  # At least 1 file for training
    train_split = files_shuffled[:n_train]
    valid_split = files_shuffled[n_train:]

    # If only 1 file, duplicate it to both sets (not ideal but ensures coverage)
    if len(files) == 1:
        valid_split = [files[0]]
        print(f"  ⚠️  {activity:10s} | Session {session}: Only 1 file, duplicating to both splits")

    # Copy files to new directories
    for f in train_split:
        shutil.copy2(f, NEW_TRAIN_DIR / f.name)
        train_count += 1

    for f in valid_split:
        shutil.copy2(f, NEW_VALID_DIR / f.name)
        valid_count += 1

    print(f"  {activity:10s} | Session {session}: {len(train_split)} train, {len(valid_split)} valid")

print(f"\n✅ Split Complete!")
print(f"  - Training files: {train_count}")
print(f"  - Validation files: {valid_count}")
print(f"\n📁 New directories created:")
print(f"  - {NEW_TRAIN_DIR.absolute()}")
print(f"  - {NEW_VALID_DIR.absolute()}")

print(f"\n⚠️  IMPORTANT: Update your notebook to use the new directories:")
print(f'     TRAIN_DATA_PATH = Path("small_train_data_csi_fixed")')
print(f'     VALID_DATA_PATH = Path("small_valid_data_csi_fixed")')
