import os
import glob
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
import joblib

# --- Config ---
CHUNK_SIZE = 500
DATA_PATH = './data'
PCA_COMPONENTS = 100

# --- Step 1: Load CSVs ---
def load_data(path=DATA_PATH):
    all_files = glob.glob(os.path.join(path, "*.csv"))
    data = []
    labels = []

    for file in all_files:
        if os.path.getsize(file) == 0:
            print(f"Skipping empty file: {file}")
            continue
        try:
            df = pd.read_csv(file)
            if 'packet_length' not in df.columns or 'label' not in df.columns:
                print(f"Skipping invalid format file: {file}")
                continue

            for i in range(0, len(df), CHUNK_SIZE):
                chunk = df['packet_length'].values[i:i+CHUNK_SIZE]
                if len(chunk) == CHUNK_SIZE:
                    data.append(chunk)
                    labels.append(df['label'].iloc[0])
        except Exception as e:
            print(f"Error reading {file}: {e}")

    return np.array(data), np.array(labels)


# --- Step 2: Load + Encode Labels ---
X, y = load_data()
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Optional: Print label mapping
print("Label mapping:", dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))))

# --- Step 3: Train PCA + XGBoost Classifier ---
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.5, random_state=47)

pca = PCA(n_components=PCA_COMPONENTS)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

clf = XGBClassifier(
    n_estimators=1000,
    learning_rate=0.2,
    max_depth=7,
    objective='multi:softmax',
    num_class=len(np.unique(y_encoded)),
    eval_metric='mlogloss',
    use_label_encoder=False,
    random_state=57
)

clf.fit(X_train_pca, y_train)

# --- Step 4: Save Models ---
joblib.dump(pca, 'pca.pkl')
joblib.dump(clf, 'model.pkl')
joblib.dump(label_encoder, 'label_encoder.pkl')

print("\n✅ Models saved: pca.pkl, model.pkl, label_encoder.pkl")

# --- Step 5: Evaluate ---
y_pred = clf.predict(X_test_pca)
print("\n📊 Classification Report:\n")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
