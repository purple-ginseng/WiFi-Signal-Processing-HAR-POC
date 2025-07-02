# train_rssi_model.py

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

# --- Load and Prepare Data ---
def load_rssi_data():
    files = glob.glob(os.path.join(DATA_DIR, "wifisignal_data_*.csv"))
    data, labels = [], []

    for f in files:
        df = pd.read_csv(f)
        pkt_cols = [c for c in df.columns if c.startswith("pkt")]
        if not pkt_cols:
            continue
        try:
            X = df[pkt_cols].values.astype(float)
            y = df["label"].values
            data.append(X)
            labels.append(y)
        except Exception as e:
            print(f"[ERROR] Skipped {f}: {e}")

    X = np.vstack(data)
    y = np.hstack(labels)
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
def train():
    X, y = load_rssi_data()
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=TEST_SIZE, random_state=42)

    if USE_PCA:
        print("Applying PCA...")
        pca = PCA(n_components=PCA_COMPONENTS)
        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)
        input_shape = (PCA_COMPONENTS,)
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
    train()
