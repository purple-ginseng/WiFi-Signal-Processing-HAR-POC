# train_rssi_model_optimized.py

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, Input
from tensorflow.keras.mixed_precision import set_global_policy

# --- GPU + AMP Optimization ---
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"[INFO] Using GPU: {[gpu.name for gpu in gpus]}")
    except RuntimeError as e:
        print(f"[ERROR] GPU initialization failed: {e}")
else:
    print("[WARNING] No GPU detected. Using CPU fallback.")

# Enable Mixed Precision and XLA
set_global_policy('mixed_float16')
tf.config.optimizer.set_jit(True)

# --- Configuration ---
DATA_DIR = "./data"
PCA_COMPONENTS = 35
TEST_SIZE = 0.4
EPOCHS = 50
BATCH_SIZE = 16
USE_PCA = True

# --- Load and Prepare Data ---
def load_rssi_data():
    files = glob.glob(os.path.join(DATA_DIR, "wifisignal_data_*.csv"))
    data, labels = [], []

    if not files:
        print(f"[ERROR] No CSV files found in {DATA_DIR}.")
        return None, None

    for f in files:
        try:
            df = pd.read_csv(f)
            pkt_cols = [c for c in df.columns if c.startswith("pkt")]
            if not pkt_cols:
                print(f"[WARNING] No 'pkt' columns found in {f}. Skipping.")
                continue
            X = df[pkt_cols].values.astype(float)
            y = df["label"].values
            data.append(X)
            labels.append(y)
        except Exception as e:
            print(f"[ERROR] Skipped {f}: {e}")

    if not data:
        return None, None

    X = np.vstack(data)
    y = np.hstack(labels)
    return X, y

# --- Model Definition ---
def build_cnn_lstm(input_shape, num_classes):
    input_tensor = Input(shape=input_shape, name="input_rssi_sequence")
    x = layers.Reshape((input_shape[0], 1), name="reshape_for_conv1d")(input_tensor)
    x = layers.Conv1D(64, 3, activation='relu', name="conv1d_layer")(x)
    x = layers.MaxPooling1D(2, name="maxpooling1d_layer")(x)
    x = layers.Dropout(0.5, name="dropout_conv")(x)
    x = layers.LSTM(64, name="lstm_layer")(x)
    x = layers.Dropout(0.5, name="dropout_lstm")(x)
    x = layers.Dense(64, activation='relu', name="dense_hidden_layer")(x)
    x = layers.Dropout(0.5, name="dropout_dense")(x)
    output_tensor = layers.Dense(num_classes, activation='softmax', dtype='float32', name="output_layer")(x)

    model = models.Model(inputs=input_tensor, outputs=output_tensor, name="CNN_LSTM_RSSI_Classifier")
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# --- Train and Evaluate ---
def train():
    X, y = load_rssi_data()
    if X is None:
        return

    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    num_classes = len(le.classes_)
    print(f"Detected {num_classes} classes: {le.classes_}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=TEST_SIZE, random_state=42, stratify=y_enc)

    scaler = StandardScaler()

    if USE_PCA:
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        pca = PCA(n_components=PCA_COMPONENTS)
        X_train_processed = pca.fit_transform(X_train_scaled)
        X_test_processed = pca.transform(X_test_scaled)
        input_shape = (PCA_COMPONENTS,)

        plt.figure(figsize=(10, 6))
        plt.plot(np.cumsum(pca.explained_variance_ratio_))
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Explained Variance')
        plt.title('PCA Explained Variance Ratio')
        plt.grid(True)
        plt.savefig('pca_explained_variance.png')
    else:
        X_train_processed = scaler.fit_transform(X_train)
        X_test_processed = scaler.transform(X_test)
        input_shape = (X.shape[1],)

    model = build_cnn_lstm(input_shape, num_classes=num_classes)
    model.summary()

    train_dataset = tf.data.Dataset.from_tensor_slices((X_train_processed, y_train))
    val_dataset = tf.data.Dataset.from_tensor_slices((X_test_processed, y_test))
    train_dataset = train_dataset.shuffle(1024).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    val_dataset = val_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    cb = [
        callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=3, min_lr=1e-6),
        callbacks.ModelCheckpoint(filepath='best_model.keras', monitor='val_accuracy', save_best_only=True),
        callbacks.TensorBoard(log_dir="./logs")
    ]

    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=EPOCHS,
        callbacks=cb,
        verbose=2
    )

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Loss')
    plt.legend(); plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.title('Accuracy')
    plt.legend(); plt.grid(True)
    plt.tight_layout()
    plt.savefig('training_history.png')

    best_model = models.load_model('best_model.keras')
    loss, acc = best_model.evaluate(val_dataset, verbose=0)
    print(f"Test Loss: {loss:.4f}, Test Accuracy: {acc:.4f}")

    preds = np.argmax(best_model.predict(val_dataset), axis=1)
    y_true = np.concatenate([y for x, y in val_dataset], axis=0)

    cm = confusion_matrix(y_true, preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
    plt.xlabel("Predicted"); plt.ylabel("True"); plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")

    report = classification_report(y_true, preds, target_names=le.classes_)
    print("\nClassification Report:\n", report)
    with open("classification_report.txt", "w") as f:
        f.write(report)

if __name__ == "__main__":
    if not os.path.exists(DATA_DIR):
        print(f"Error: Data directory '{DATA_DIR}' not found.")
    else:
        train()