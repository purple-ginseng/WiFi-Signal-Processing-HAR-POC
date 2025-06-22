import os
import glob
import numpy as np
import pandas as pd
import joblib
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

# --- Config ---
DATA_PATH = './data'
FEATURE_COLUMNS = ['packet_length']
CHUNK_SIZE = 300
PCA_COMPONENTS = 50
TEST_SIZE = 0.3
EPOCHS = 50
BATCH_SIZE = 32

def load_data(path, chunk_size):
    all_files = glob.glob(os.path.join(path, "*.csv"))
    data, labels = [], []

    for file in all_files:
        if os.path.getsize(file) == 0:
            continue
        try:
            df = pd.read_csv(file)
            if not set(FEATURE_COLUMNS + ['label']).issubset(df.columns):
                continue

            for i in range(0, len(df) - chunk_size + 1, chunk_size // 2):
                chunk = df[FEATURE_COLUMNS].iloc[i:i + chunk_size].values.flatten()
                if len(chunk) == chunk_size * len(FEATURE_COLUMNS):
                    data.append(chunk)
                    labels.append(df['label'].iloc[0])
        except Exception as e:
            print(f"Error reading {file}: {e}")
    return np.array(data), np.array(labels)


def plot_confusion_matrix(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig("confusion_matrix_tf.png")
    plt.close()


# --- Load Data ---
X, y = load_data(DATA_PATH, CHUNK_SIZE)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
num_classes = len(np.unique(y_encoded))

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=TEST_SIZE, random_state=42)

pca = PCA(n_components=PCA_COMPONENTS)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# --- Model ---
model = models.Sequential([
    layers.Input(shape=(PCA_COMPONENTS,)),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(64, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

cb = [
    callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    callbacks.TensorBoard(log_dir='./logs/tf_model')
]

model.fit(
    X_train_pca, y_train,
    validation_split=0.2,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=cb,
    verbose=2
)

# --- Evaluate ---
y_pred = model.predict(X_test_pca)
y_pred_classes = np.argmax(y_pred, axis=1)

print("\nClassification Report (TF):")
print(classification_report(y_test, y_pred_classes, target_names=label_encoder.classes_))

plot_confusion_matrix(y_test, y_pred_classes, label_encoder.classes_)

# --- Save ---
model.save('tf_model.keras')
joblib.dump(pca, 'pca_tf.pkl')
joblib.dump(label_encoder, 'label_encoder_tf.pkl')
print("\n✅ TensorFlow model, PCA, and label encoder saved.")