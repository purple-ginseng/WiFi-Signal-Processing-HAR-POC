import os
import glob
import argparse
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier, callback
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import csv
import time
from datetime import datetime

# --- Default Features ---
DEFAULT_FEATURE_COLUMNS = ['packet_length']  # extendable to rssi, snr, etc.


def load_data(path, chunk_size, feature_columns):
    all_files = glob.glob(os.path.join(path, "*.csv"))
    data, labels = [], []

    for file in all_files:
        if os.path.getsize(file) == 0:
            continue
        try:
            df = pd.read_csv(file)
            if not set(feature_columns + ['label']).issubset(df.columns):
                continue

            for i in range(0, len(df) - chunk_size + 1, chunk_size // 2):
                chunk = df[feature_columns].iloc[i:i + chunk_size].values.flatten()
                if len(chunk) == chunk_size * len(feature_columns):
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
    plt.savefig("confusion_matrix.png")
    plt.close()


def save_csv_log(log, filename="training_log.csv"):
    keys = log[0].keys()
    with open(filename, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(log)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./data')
    parser.add_argument('--chunk_size', type=int, default=300)
    parser.add_argument('--pca_components', type=int, default=50)
    parser.add_argument('--test_size', type=float, default=0.3)
    parser.add_argument('--learning_rate', type=float, default=0.1)
    parser.add_argument('--max_depth', type=int, default=7)
    parser.add_argument('--n_estimators', type=int, default=1000)
    parser.add_argument('--enable_confusion_matrix', action='store_true')
    parser.add_argument('--enable_csv_log', action='store_true')
    parser.add_argument('--enable_tensorboard', action='store_true')
    parser.add_argument('--use_grid_search', action='store_true')
    parser.add_argument('--features', type=str, default=','.join(DEFAULT_FEATURE_COLUMNS), help='Comma-separated feature list')
    args = parser.parse_args()

    feature_columns = args.features.split(',')
    print("\n📦 Loading data with features:", feature_columns)
    X, y = load_data(args.data_path, args.chunk_size, feature_columns)
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=args.test_size, random_state=47)

    print("📊 Performing PCA...")
    pca = PCA(n_components=args.pca_components)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    log = []
    logdir = f"runs/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    tb_callback = callback.TrainingCallback() if not args.enable_tensorboard else callback.TensorBoard(logdir=logdir)

    if args.use_grid_search:
        print("🔍 Running Grid Search...")
        param_grid = {
            'learning_rate': [0.05, 0.1, 0.2],
            'max_depth': [5, 7, 9],
            'n_estimators': [300, 500, 1000]
        }
        grid = GridSearchCV(
            XGBClassifier(
                objective='multi:softmax',
                num_class=len(np.unique(y_encoded)),
                eval_metric='mlogloss',
                use_label_encoder=False,
                random_state=57
            ),
            param_grid,
            scoring='accuracy',
            cv=3,
            verbose=2
        )
        grid.fit(X_train_pca, y_train)
        clf = grid.best_estimator_
        print("✅ Best params:", grid.best_params_)
    else:
        clf = XGBClassifier(
            n_estimators=args.n_estimators,
            learning_rate=args.learning_rate,
            max_depth=args.max_depth,
            objective='multi:softmax',
            num_class=len(np.unique(y_encoded)),
            eval_metric='mlogloss',
            use_label_encoder=False,
            random_state=57
        )

        print("🚀 Training...")
        clf.fit(
            X_train_pca, y_train,
            eval_set=[(X_test_pca, y_test)],
            early_stopping_rounds=20,
            callbacks=[tb_callback] if args.enable_tensorboard else None,
            verbose=True
        )

    print("💾 Saving models...")
    joblib.dump(pca, 'pca.pkl')
    joblib.dump(clf, 'model.pkl')
    joblib.dump(label_encoder, 'label_encoder.pkl')

    print("📈 Evaluating...")
    y_pred = clf.predict(X_test_pca)
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

    if args.enable_confusion_matrix:
        print("🧩 Saving confusion matrix...")
        plot_confusion_matrix(y_test, y_pred, label_encoder.classes_)

    if args.enable_csv_log:
        print("📄 Saving training log...")
        log.append({
            "chunk_size": args.chunk_size,
            "pca_components": args.pca_components,
            "test_size": args.test_size,
            "learning_rate": getattr(clf, 'learning_rate', args.learning_rate),
            "max_depth": getattr(clf, 'max_depth', args.max_depth),
            "n_estimators": getattr(clf, 'n_estimators', args.n_estimators),
            "accuracy": np.mean(y_pred == y_test)
        })
        save_csv_log(log)