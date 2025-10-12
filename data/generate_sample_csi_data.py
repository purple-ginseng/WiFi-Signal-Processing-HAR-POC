#!/usr/bin/env python3
"""
Sample CSI Data Generator for Testing WiFi HAR Pipeline

This script generates synthetic CSI data with realistic characteristics for testing
the CSI-based WiFi HAR pipeline. Real CSI data should be collected using:
- Intel WiFi Link 5300 NIC with Linux CSI Tool
- Atheros AR9580 with CSI Tool
- ESP32-S3 with CSI example code

Generated data format:
- amplitude_0 to amplitude_29: CSI amplitude per subcarrier
- phase_0 to phase_29: CSI phase per subcarrier (radians)
- label: Activity class
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple

# Configuration
NUM_SUBCARRIERS = 30
ACTIVITIES = ['falling', 'sitting', 'sleeping', 'standing', 'walking']
SAMPLES_PER_ACTIVITY = 2000

# Output directories
TRAIN_DIR = Path("small_train_data_csi")
VALID_DIR = Path("small_valid_data_csi")


def generate_activity_csi(
    activity: str,
    num_samples: int = 2000,
    num_subcarriers: int = 30
) -> pd.DataFrame:
    """
    Generate synthetic CSI data with activity-specific characteristics.

    Activity patterns:
    - Falling: High amplitude variance, rapid phase changes
    - Sitting: Low variance, minimal phase drift
    - Sleeping: Very low variance, almost static
    - Standing: Low variance with occasional micro-movements
    - Walking: Periodic patterns in both amplitude and phase
    """
    np.random.seed(hash(activity) % 2**32)

    data = {}

    # Activity-specific parameters
    if activity == 'falling':
        amp_base = 0.8
        amp_variance = 0.4
        phase_velocity = 0.5  # rad/sample
        phase_noise = 0.3
    elif activity == 'sitting':
        amp_base = 0.6
        amp_variance = 0.05
        phase_velocity = 0.02
        phase_noise = 0.05
    elif activity == 'sleeping':
        amp_base = 0.5
        amp_variance = 0.02
        phase_velocity = 0.01
        phase_noise = 0.02
    elif activity == 'standing':
        amp_base = 0.65
        amp_variance = 0.08
        phase_velocity = 0.03
        phase_noise = 0.06
    elif activity == 'walking':
        amp_base = 0.7
        amp_variance = 0.15
        phase_velocity = 0.2
        phase_noise = 0.1
    else:
        raise ValueError(f"Unknown activity: {activity}")

    # Generate CSI for each subcarrier
    for i in range(num_subcarriers):
        # Amplitude: activity-specific + multipath fading
        if activity == 'walking':
            # Periodic pattern for walking (step cycle)
            t = np.arange(num_samples)
            periodic = 0.1 * np.sin(2 * np.pi * t / 20)  # ~20 samples per step
            amplitude = amp_base + periodic + np.random.randn(num_samples) * amp_variance
        else:
            amplitude = amp_base + np.random.randn(num_samples) * amp_variance

        # Ensure amplitude is positive
        amplitude = np.clip(amplitude, 0.1, 2.0)

        # Phase: cumulative sum (Doppler shift) + noise
        if activity == 'walking':
            # Periodic phase changes for walking
            t = np.arange(num_samples)
            phase_trend = phase_velocity * np.sin(2 * np.pi * t / 20)
        else:
            phase_trend = phase_velocity * np.ones(num_samples)

        phase_changes = phase_trend + np.random.randn(num_samples) * phase_noise
        phase = np.cumsum(phase_changes)

        # Wrap phase to [-π, π]
        phase = np.arctan2(np.sin(phase), np.cos(phase))

        # Store in dataframe
        data[f'amplitude_{i}'] = amplitude
        data[f'phase_{i}'] = phase

    # Add label
    data['label'] = activity

    df = pd.DataFrame(data)
    return df


def generate_dataset(output_dir: Path, train_split: float = 0.8):
    """
    Generate complete CSI dataset with train/validation split.
    """
    print(f"Generating CSI dataset in: {output_dir}")
    output_dir.mkdir(exist_ok=True, parents=True)

    for activity in ACTIVITIES:
        print(f"  Generating {activity}...")

        # Generate data
        df = generate_activity_csi(activity, SAMPLES_PER_ACTIVITY, NUM_SUBCARRIERS)

        # Save to CSV
        filename = f"csi_data_{activity}.csv"
        output_path = output_dir / filename
        df.to_csv(output_path, index=False)

        print(f"    Saved {len(df)} samples to {filename}")

    print(f"✓ Dataset generation complete!")


def main():
    """
    Generate both training and validation CSI datasets.
    """
    print("=" * 60)
    print("CSI Sample Data Generator for WiFi HAR")
    print("=" * 60)
    print()
    print("⚠️  NOTE: This generates SYNTHETIC data for testing.")
    print("   For real experiments, collect actual CSI data using:")
    print("   - Intel WiFi Link 5300 + Linux CSI Tool")
    print("   - Atheros AR9580 + CSI Tool")
    print("   - ESP32-S3 with CSI example code")
    print()

    # Generate training data
    print("\n=== Generating Training Data ===")
    generate_dataset(TRAIN_DIR)

    # Generate validation data (different random seed)
    print("\n=== Generating Validation Data ===")
    np.random.seed(42)  # Different seed for validation
    generate_dataset(VALID_DIR)

    print("\n" + "=" * 60)
    print("✓ All datasets generated successfully!")
    print(f"  Training data: {TRAIN_DIR.absolute()}")
    print(f"  Validation data: {VALID_DIR.absolute()}")
    print(f"  Activities: {', '.join(ACTIVITIES)}")
    print(f"  Samples per activity: {SAMPLES_PER_ACTIVITY}")
    print(f"  Subcarriers: {NUM_SUBCARRIERS}")
    print("=" * 60)


if __name__ == "__main__":
    main()
