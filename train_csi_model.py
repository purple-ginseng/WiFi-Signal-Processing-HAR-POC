# train_csi_model.py

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

# --- Configuration ---
DATA_DIR = "./data"
PCA_COMPONENTS = 50
TEST_SIZE = 0.3
EPOCHS = 30
BATCH_SIZE = 32
USE_PCA = True  # Toggle to False for raw input
USE_MAGNITUDE = True  # Toggle to False for using [I, Q] instead

# --- Load and Prepare CSI Data ---
def load_signal_data(prefix):
    pattern = os.path.join(DATA_DIR, f"{prefix}_*.csv")
    files = glob.glob(pattern)
    data, labels = [], []

    for f in files:
        df = pd.read_csv(f)
        if not {"I", "Q", "magnitude"}.issubset(df.columns):
            print(f"[WARN] Skipping invalid file {f}")
            continue

        try:
            inferred_label = os.path.basename(f).split("_")[2]  # e.g., esp32_csi_sitting_2025.csv → "sitting"
            df["label"] = inferred_label

            grouped = df.groupby("timestamp")
            for ts, group in grouped:
                if USE_MAGNITUDE:
                    features = group["magnitude"].values
                else:
                    features = group[["I", "Q"]].values.flatten()

                if len(features) > 0:
                    label = group["label"].iloc[0]
                    data.append(features)
                    labels.append(label)
        except Exception as e:
            print(f"[ERROR] Skipped {f}: {e}")

    X = np.array(data)
    y = np.array(labels)
    return X, y

# --- Model Definition ---
def build_cnn_lstm(input_shape, num_classes):
    model = models.Sequential([
        layers.Reshape((input_shape[0], 1), input_shape=input_shape),
        layers.Conv1D(64, 3, activation='relu'),
        layers.MaxPooling1D(2),
        layers.LSTM(64),
        layers.Dense(64, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# --- Train and Evaluate ---
def train(prefix):
    X, y = load_signal_data(prefix)

    if len(X) == 0:
        print("[ERROR] No valid samples found. Check your CSV files and ensure they contain timestamped CSI rows with valid 'magnitude' or 'I/Q'.")
        return

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=TEST_SIZE, random_state=42)

    if USE_PCA:
        print("Applying PCA...")
        pca = PCA(n_components=min(PCA_COMPONENTS, X_train.shape[1]))
        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)
        input_shape = (X_train.shape[1],)
    else:
        input_shape = (X.shape[1],)

    model = build_cnn_lstm(input_shape, num_classes=len(le.classes_))

    cb = [callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)]
    model.fit(X_train, y_train, validation_split=0.2, epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=cb, verbose=2)

    preds = np.argmax(model.predict(X_test), axis=1)
    cm = confusion_matrix(y_test, preds)

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=le.classes_, yticklabels=le.classes_)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()

    print("\nClassification Report:\n")
    print(classification_report(y_test, preds, target_names=le.classes_))

if __name__ == "__main__":
    train(prefix="esp32_csi")