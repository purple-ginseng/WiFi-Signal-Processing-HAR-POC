"""
WiFi CSI-based Human Activity Recognition with Range-Doppler Velocity Features
PyTorch implementation with CPU support for macOS
"""

import pandas as pd
from pathlib import Path
from typing import List, Optional
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from torch import nn
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.signal import detrend


def merge_csv(
    folder_path: Optional[Path] = None,
    file_paths: Optional[List[Path]] = None
) -> pd.DataFrame:
    """
    Merges CSV files from either a specified folder or a list of file paths.

    Args:
        folder_path: The path to the folder containing CSV files to merge.
        file_paths: A list of specific CSV file paths to merge.

    Returns:
        A single pandas DataFrame containing the merged data.

    Raises:
        ValueError: If neither or both `folder_path` and `file_paths` are provided,
                    or if no CSV files are found.
    """
    # 1. Input Validation (Guard Clauses)
    if (folder_path is None and file_paths is None) or \
       (folder_path is not None and file_paths is not None):
        raise ValueError("Provide either 'folder_path' OR 'file_paths', not both/neither.")

    # 2. Determine the list of files based on the mode
    if folder_path:
        files = list(folder_path.glob("*.csv"))
    else:
        files = file_paths

    if not files:
        raise ValueError("No CSV files found to merge.")

    # 3. Efficient Merging Logic
    df_list = (pd.read_csv(file) for file in files)
    return pd.concat(df_list, ignore_index=True)


def extract_velocity_features(df):
    """
    Extract velocity-based features from CSI data.
    Adapts to available columns in the dataframe.
    """
    print(f"Input columns: {df.columns.tolist()[:10]}... ({len(df.columns)} total)")

    velocity_features = pd.DataFrame()

    # Get all numeric columns except label
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'label' in numeric_cols:
        numeric_cols.remove('label')

    if len(numeric_cols) == 0:
        print("Warning: No numeric columns found for velocity extraction")
        return df

    # 1. First-order differences (velocity proxy)
    for col in numeric_cols[:10]:
        velocity_features[f'vel_{col}'] = df[col].diff().fillna(0).replace([np.inf, -np.inf], 0)

    # 2. Second-order differences (acceleration proxy)
    for col in numeric_cols[:5]:
        velocity_features[f'acc_{col}'] = df[col].diff().diff().fillna(0).replace([np.inf, -np.inf], 0)

    # 3. Rolling statistics (motion patterns)
    window = min(10, max(2, len(df) // 100))
    for col in numeric_cols[:5]:
        roll_std = df[col].rolling(window=window, min_periods=1).std()
        velocity_features[f'roll_std_{col}'] = roll_std.fillna(0).replace([np.inf, -np.inf], 0)

        roll_max = df[col].rolling(window=window, min_periods=1).max()
        roll_min = df[col].rolling(window=window, min_periods=1).min()
        roll_range = (roll_max - roll_min).fillna(0).replace([np.inf, -np.inf], 0)
        velocity_features[f'roll_range_{col}'] = roll_range

    # 4. Global velocity statistics
    diff_abs = df[numeric_cols].diff().abs()
    velocity_features['overall_velocity'] = diff_abs.mean(axis=1).fillna(0).replace([np.inf, -np.inf], 0)

    diff2_abs = df[numeric_cols].diff().diff().abs()
    velocity_features['overall_acceleration'] = diff2_abs.mean(axis=1).fillna(0).replace([np.inf, -np.inf], 0)

    signal_std = df[numeric_cols].std(axis=1)
    velocity_features['signal_volatility'] = signal_std.fillna(0).replace([np.inf, -np.inf], 0)

    print(f"Created {len(velocity_features.columns)} velocity features")

    # Verify no NaN values
    nan_count = velocity_features.isna().sum().sum()
    inf_count = np.isinf(velocity_features.select_dtypes(include=[np.number])).sum().sum()

    if nan_count > 0:
        print(f"Warning: {nan_count} NaN values found, filling with 0")
        velocity_features = velocity_features.fillna(0)

    if inf_count > 0:
        print(f"Warning: {inf_count} Inf values found, replacing with 0")
        velocity_features = velocity_features.replace([np.inf, -np.inf], 0)

    # Concatenate with original dataframe
    df_with_velocity = pd.concat([df, velocity_features], axis=1)

    print(f"Final NaN count: {df_with_velocity.isna().sum().sum()}")

    return df_with_velocity


class SimpleMLP(nn.Module):
    """
    Simplified MLP for better generalization on tabular data.
    Much smaller capacity to prevent overfitting.
    """
    def __init__(self, input_size, num_classes):
        super(SimpleMLP, self).__init__()

        # Simpler architecture: 3 hidden layers with decreasing size
        self.fc1 = nn.Linear(input_size, 128)  # Reduced from 256
        self.bn1 = nn.BatchNorm1d(128)
        self.dropout1 = nn.Dropout(0.5)  # Increased from 0.3

        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.dropout2 = nn.Dropout(0.5)

        self.fc3 = nn.Linear(64, 32)  # Additional bottleneck
        self.bn3 = nn.BatchNorm1d(32)
        self.dropout3 = nn.Dropout(0.4)

        self.fc4 = nn.Linear(32, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = torch.relu(x)
        x = self.dropout2(x)

        x = self.fc3(x)
        x = self.bn3(x)
        x = torch.relu(x)
        x = self.dropout3(x)

        return self.fc4(x)


def plot_confusion_matrix(cm, categories, title="Confusion Matrix with Counts and Percentages"):
    """
    Plot confusion matrix with both counts and normalized percentages in each cell.
    """
    cm = np.array(cm, dtype=int)
    cm_normalized = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]

    labels = np.array([
        [f"{count}\n({perc:.2%})" if cm[i, j] != 0 else ""
         for j, (count, perc) in enumerate(zip(row, row_norm))]
        for i, (row, row_norm) in enumerate(zip(cm, cm_normalized))
    ])

    plt.figure(figsize=(9,7))
    ax = sns.heatmap(cm, annot=labels, fmt="", cmap="Blues", cbar=True,
                     xticklabels=categories, yticklabels=categories,
                     annot_kws={"size":10, "weight":"bold"})

    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    plt.tight_layout()
    plt.show()


def main():
    # Configuration
    TRAIN_DATA_PATH = Path("small_train_data")
    VALID_DATA_PATH = Path("small_valid_data")
    BATCH_SIZE = 256
    EPOCHS = 50  # Increase epochs, rely on early stopping
    EARLY_STOPPING_PATIENCE = 15  # More patience for slower learning
    MODEL_PATH = Path("modelcheckpoints/best_model.pt")

    # WiFi parameters for velocity calculation
    CARRIER_FREQ = 5.8e9  # 5.8 GHz WiFi
    SPEED_OF_LIGHT = 3e8  # m/s
    WAVELENGTH = SPEED_OF_LIGHT / CARRIER_FREQ

    # Auto-detect device: CUDA if available, otherwise CPU (macOS compatible)
    if torch.cuda.is_available():
        device = "cuda"
        print(f"Using device: {device} (GPU)")
    elif torch.backends.mps.is_available():
        device = "mps"
        print(f"Using device: {device} (Apple Silicon GPU)")
    else:
        device = "cpu"
        print(f"Using device: {device} (CPU)")

    # Load TRAINING data
    print("\n=== Loading Training Data ===")
    df_train = merge_csv(folder_path=TRAIN_DATA_PATH)
    print(f"Training data shape: {df_train.shape}")
    print(f"Training labels: {df_train['label'].value_counts().to_dict()}")

    # Load VALIDATION data
    print("\n=== Loading Validation Data ===")
    df_valid = merge_csv(folder_path=VALID_DATA_PATH)
    print(f"Validation data shape: {df_valid.shape}")
    print(f"Validation labels: {df_valid['label'].value_counts().to_dict()}")

    # Extract velocity features for BOTH datasets
    print("\n=== Extracting Velocity Features for Training Data ===")
    df_train = extract_velocity_features(df_train)
    print(f"Enhanced training shape: {df_train.shape}")

    print("\n=== Extracting Velocity Features for Validation Data ===")
    df_valid = extract_velocity_features(df_valid)
    print(f"Enhanced validation shape: {df_valid.shape}")

    # Preprocessing pipeline
    print("\n=== Building Preprocessing Pipeline ===")
    TARGET_COLUMN = "label"
    FEATURE_COLUMNS = [col for col in df_train.columns if col != TARGET_COLUMN]

    le = ColumnTransformer(
        transformers=[("encoder", OrdinalEncoder(), [TARGET_COLUMN])],
        remainder="passthrough",
        verbose_feature_names_out=False
    )

    scaler = ColumnTransformer(
        transformers=[("scaler", StandardScaler(), FEATURE_COLUMNS)],
        remainder="passthrough",
        verbose_feature_names_out=False
    )

    pca = ColumnTransformer(
        transformers=[("pca", PCA(), FEATURE_COLUMNS)],
        remainder="passthrough",
        verbose_feature_names_out=False
    )

    pipeline = make_pipeline(le, scaler, pca)
    pipeline.set_output(transform="pandas")

    # FIT on training data, TRANSFORM both
    print("\n=== Applying Pipeline ===")
    df_train = pipeline.fit_transform(df_train)
    df_valid = pipeline.transform(df_valid)
    print(f"Final training shape: {df_train.shape}")
    print(f"Final validation shape: {df_valid.shape}")

    # Use separate datasets (NO train_test_split!)
    print("\n=== Preparing Datasets ===")
    X_train = df_train.drop(TARGET_COLUMN, axis=1).to_numpy()
    y_train = df_train[TARGET_COLUMN].to_numpy()
    X_valid = df_valid.drop(TARGET_COLUMN, axis=1).to_numpy()
    y_valid = df_valid[TARGET_COLUMN].to_numpy()
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Validation samples: {X_valid.shape[0]}")

    # Convert to PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.int64)
    X_valid = torch.tensor(X_valid, dtype=torch.float32)
    y_valid = torch.tensor(y_valid, dtype=torch.int64)

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_valid, y_valid), batch_size=BATCH_SIZE)

    # Initialize model
    print("\n=== Initializing Model ===")
    model = SimpleMLP(
        input_size=X_train.shape[1],
        num_classes=len(le.named_transformers_["encoder"].categories_[0])
    )
    model = model.to(device)

    # Class weights for imbalanced data
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train.numpy()
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

    # Training setup with stronger regularization
    criterion = nn.CrossEntropyLoss(label_smoothing=0.15, weight=class_weights)  # Increased label smoothing
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=5e-3)  # Higher weight decay (50x)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True)

    # Training loop
    print("\n=== Training Model ===")
    patience = EARLY_STOPPING_PATIENCE
    best_val_acc = 0.0  # Track best validation accuracy

    for epoch in range(EPOCHS):
        # === Training Phase ===
        model.train()
        running_loss = 0.0
        progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False)

        for xb, yb in progress:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()

            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)

        # === Validation Phase ===
        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        y_preds, y_true = [], []

        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                preds = model(xb)
                loss = criterion(preds, yb)
                val_loss += loss.item()
                correct += (preds.argmax(dim=1) == yb).sum().item()
                total += yb.size(0)
                y_preds.extend(preds.argmax(dim=1).cpu().numpy())
                y_true.extend(yb.cpu().numpy())

        val_loss /= len(val_loader)
        val_acc = correct / total

        # Update learning rate based on validation accuracy
        scheduler.step(val_acc)

        print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # Early stopping based on VALIDATION ACCURACY (not loss)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_PATH)
            patience = EARLY_STOPPING_PATIENCE
            print(f"  → New best validation accuracy: {best_val_acc:.4f}, model saved!")
        else:
            patience -= 1
            if patience == 0:
                print(f"Early stopping triggered. Best val acc: {best_val_acc:.4f}")
                break

    print(f"\n=== Training Complete ===")
    print(f"Best Validation Accuracy: {best_val_acc:.4f}")

    # Save final model
    print("\n=== Saving Final Model ===")
    model.eval()
    torch.save(model.state_dict(), "modelcheckpoints/TrainTestSplit.pt")

    # Generate confusion matrices for BOTH training and validation
    categories = le.named_transformers_['encoder'].categories_[0]
    model.to(device)

    # Training confusion matrix
    print("\n=== Computing Training Confusion Matrix ===")
    cm_train = np.zeros((len(categories), len(categories)), dtype=int)
    for X_batch, y_batch in tqdm(train_loader, desc="Training confusion matrix"):
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        pred = model(X_batch).argmax(dim=1).cpu()
        y_batch = y_batch.cpu()
        cm_train += confusion_matrix(y_batch, pred, labels=np.arange(len(categories)))

    print("\nTraining Confusion Matrix:")
    print(pd.DataFrame(cm_train, columns=categories, index=categories))

    # Validation confusion matrix
    print("\n=== Computing Validation Confusion Matrix ===")
    cm_valid = np.zeros((len(categories), len(categories)), dtype=int)
    for X_batch, y_batch in tqdm(val_loader, desc="Validation confusion matrix"):
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        pred = model(X_batch).argmax(dim=1).cpu()
        y_batch = y_batch.cpu()
        cm_valid += confusion_matrix(y_batch, pred, labels=np.arange(len(categories)))

    print("\nValidation Confusion Matrix:")
    print(pd.DataFrame(cm_valid, columns=categories, index=categories))

    # Plot both confusion matrices
    plot_confusion_matrix(cm_train, categories, title="Confusion Matrix on Training Data")
    plot_confusion_matrix(cm_valid, categories, title="Confusion Matrix on Validation Data")

    print("\n=== Training Complete ===")


if __name__ == "__main__":
    main()
