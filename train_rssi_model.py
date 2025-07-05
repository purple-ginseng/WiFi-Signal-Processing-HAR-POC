# train_rssi_model.py

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
from tensorflow.keras import layers, models, callbacks, Input # Added Input for functional API

# --- Configuration ---
DATA_DIR = "./data"
PCA_COMPONENTS = 35 # Based on your PCA Explained Variance plot, this captures ~90-95% variance
TEST_SIZE = 0.4 # Changed to 0.4 for 60% training, 40% testing split
EPOCHS = 50 # Increased epochs as EarlyStopping will handle stopping
BATCH_SIZE = 16 # Keep this for now, but can be adjusted if training is too slow or unstable
USE_PCA = True  # Toggle to False for raw input

# --- Load and Prepare Data ---
def load_rssi_data():
    files = glob.glob(os.path.join(DATA_DIR, "wifisignal_data_*.csv"))
    data, labels = [], []

    if not files:
        print(f"[ERROR] No CSV files found in {DATA_DIR}. Please ensure your data is in this directory.")
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
        except KeyError:
            print(f"[ERROR] Skipped {f}: Missing 'label' column.")
        except ValueError as ve:
            print(f"[ERROR] Skipped {f}: Data conversion error (e.g., non-numeric values in 'pkt' columns): {ve}")
        except Exception as e:
            print(f"[ERROR] Skipped {f}: An unexpected error occurred: {e}")

    if not data:
        print("[ERROR] No valid data could be loaded. Check your CSV files and 'pkt' column names.")
        return None, None

    X = np.vstack(data)
    y = np.hstack(labels)
    return X, y

# --- Model Definition ---
def build_cnn_lstm(input_shape, num_classes):
    # Using Keras Functional API for more explicit input definition and flexibility
    input_tensor = Input(shape=input_shape, name="input_rssi_sequence")

    # Reshape for Conv1D: (sequence_length, 1)
    x = layers.Reshape((input_shape[0], 1), name="reshape_for_conv1d")(input_tensor)

    x = layers.Conv1D(filters=64, kernel_size=3, activation='relu', name="conv1d_layer")(x)
    x = layers.MaxPooling1D(pool_size=2, name="maxpooling1d_layer")(x)
    x = layers.Dropout(0.5, name="dropout_conv")(x) # Increased Dropout after ConvNet

    # Flatten before LSTM if Conv1D output is not directly compatible
    # In this specific case, Conv1D output (batch, steps, features) is fine for LSTM (batch, timesteps, features)
    # However, if your Conv1D was setup differently, you might need layers.Permute or layers.Reshape

    x = layers.LSTM(64, name="lstm_layer")(x)
    x = layers.Dropout(0.5, name="dropout_lstm")(x) # Increased Dropout after LSTM

    x = layers.Dense(64, activation='relu', name="dense_hidden_layer")(x)
    x = layers.Dropout(0.5, name="dropout_dense")(x) # Increased Dropout here too

    output_tensor = layers.Dense(num_classes, activation='softmax', name="output_layer")(x)

    model = models.Model(inputs=input_tensor, outputs=output_tensor, name="CNN_LSTM_RSSI_Classifier")

    # Lower initial learning rate for better stability
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001) # Lowered learning rate
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model

# --- Train and Evaluate ---
def train():
    X, y = load_rssi_data()
    if X is None: # Handle case where no data was loaded
        return

    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    num_classes = len(le.classes_)
    print(f"Detected {num_classes} unique classes: {le.classes_}")

    # --- PATCH START: Changed TEST_SIZE to 0.4 for 60/40 split and ensured stratification ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc,
        test_size=TEST_SIZE, # Now 0.4 for 60% train, 40% test
        random_state=42,
        stratify=y_enc # Ensures that each 'activity' (class) is proportionally represented in train and test sets
    )
    # --- PATCH END ---

    print(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}")

    # Initialize scaler outside if/else to ensure it's always available for X_test scaling
    scaler = StandardScaler()

    if USE_PCA:
        print("Applying StandardScaler and PCA...")
        # Scale data BEFORE PCA
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        pca = PCA(n_components=PCA_COMPONENTS)
        X_train_processed = pca.fit_transform(X_train_scaled)
        X_test_processed = pca.transform(X_test_scaled)
        input_shape = (PCA_COMPONENTS,)

        # Optional: Plot explained variance ratio for PCA
        plt.figure(figsize=(10, 6))
        plt.plot(np.cumsum(pca.explained_variance_ratio_))
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Explained Variance')
        plt.title('PCA Explained Variance Ratio')
        plt.grid(True)
        plt.savefig('pca_explained_variance.png')
        print(f"PCA explained variance plot saved to pca_explained_variance.png")

    else:
        print("Applying StandardScaler (without PCA)...")
        # Scale data if not using PCA
        X_train_processed = scaler.fit_transform(X_train)
        X_test_processed = scaler.transform(X_test)
        input_shape = (X.shape[1],) # Use original feature count as input shape

    print(f"Input shape for model: {input_shape}")
    model = build_cnn_lstm(input_shape, num_classes=num_classes)
    model.summary() # Print model summary to verify layers

    # Callbacks for robust training
    cb = [
        callbacks.EarlyStopping(
            monitor='val_loss',     # Monitor validation loss
            patience=10,           # Number of epochs with no improvement after which training will be stopped.
            restore_best_weights=True, # Restore model weights from the epoch with the best monitored value.
            verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',     # Monitor validation loss
            factor=0.3,             # Factor by which the learning rate will be reduced. new_lr = lr * factor
            patience=3,            # Number of epochs with no improvement after which learning rate will be reduced.
            min_lr=0.000001,        # Lower bound on the learning rate.
            verbose=1
        ),
        callbacks.ModelCheckpoint(
            filepath='best_model.keras', # --- PATCH: Changed to .keras format ---
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=0 # Suppress individual checkpoint messages
        )
    ]

    print("\nStarting model training...")
    history = model.fit(
        X_train_processed, y_train,
        validation_split=0.2, # Still using split from training data for validation during fit
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=cb,
        verbose=2 # Show per-epoch results
    )

    # Plot training history
    plt.figure(figsize=(12, 5))

    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('training_history.png')
    print("Training history plots saved to training_history.png")
    plt.show() # Display plots after saving

    print("\n--- Evaluating on Test Set ---")
    # Load the best model saved by ModelCheckpoint for final evaluation
    try:
        best_model = models.load_model('best_model.keras') # --- PATCH: Changed to load from .keras format ---
        print("Loaded best model for final evaluation.")
    except Exception as e:
        print(f"Could not load 'best_model.keras', using the last trained model. Error: {e}")
        best_model = model

    test_loss, test_accuracy = best_model.evaluate(X_test_processed, y_test, verbose=0)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")

    preds_proba = best_model.predict(X_test_processed)
    preds = np.argmax(preds_proba, axis=1)

    cm = confusion_matrix(y_test, preds)

    # Plot confusion matrix
    plt.figure(figsize=(10, 8)) # Increased figure size for better readability
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=le.classes_, yticklabels=le.classes_)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    print("Confusion matrix saved to confusion_matrix.png")
    plt.show()

    print("\nClassification Report:\n")
    print(classification_report(y_test, preds, target_names=le.classes_))
    
    # Save classification report to a text file
    report_path = 'classification_report.txt'
    with open(report_path, 'w') as f:
        f.write(classification_report(y_test, preds, target_names=le.classes_))
    print(f"Classification report saved to {report_path}")

if __name__ == "__main__":
    # Ensure the data directory exists
    if not os.path.exists(DATA_DIR):
        print(f"Error: Data directory '{DATA_DIR}' not found. Please create it and place your CSV files inside.")
    else:
        train()