import os
import glob
import logging
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime
import collections

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import autocast, GradScaler

from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight

from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import seaborn as sns
import json

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration ---
DATA_DIR = "./data"
PCA_COMPONENTS = 0.99
BATCH_SIZE = 2048
EPOCHS = 100
USE_PCA = True
EARLY_STOPPING_PATIENCE = 3
MODEL_PATH = "best_model.pt"
RUN_NAME = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

# --- Model ---
class CNNLSTMModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(CNNLSTMModel, self).__init__()
        self.conv1 = nn.Conv1d(1, 64, kernel_size=3)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool = nn.MaxPool1d(2)
        self.dropout1 = nn.Dropout(0.6)
        self.lstm = nn.LSTM(input_size=64, hidden_size=768, num_layers=2, batch_first=True, dropout=0.5, bidirectional=True)
        self.dropout2 = nn.Dropout(0.6)
        self.ln1 = nn.LayerNorm(768 * 2)
        self.fc1 = nn.Linear(768 * 2, 128)
        self.fc2 = nn.Linear(128, 64)
        self.dropout3 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.pool(x)
        x = self.dropout1(x)
        x = x.permute(0, 2, 1)
        _, (h_n, _) = self.lstm(x)
        x = torch.cat((h_n[-2], h_n[-1]), dim=1)
        x = self.ln1(x)
        x = self.dropout2(x)
        x = torch.nn.functional.gelu(self.fc1(x))
        x = torch.nn.functional.gelu(self.fc2(x))
        x = self.dropout3(x)
        return self.fc3(x)

# --- Data Loading ---
def load_data():
    logger.info("Loading data from CSV files...")
    files = glob.glob(os.path.join(DATA_DIR, "wifisignal_data_*.csv"))
    if not files:
        logger.error(f"No CSV files found in {DATA_DIR}.")
        sys.exit(1)

    all_X, all_y = [], []

    for f in files:
        try:
            df = pd.read_csv(f)
            pkt_cols = [c for c in df.columns if c.startswith("pkt")]
            if not pkt_cols or 'label' not in df.columns:
                logger.warning(f"Invalid format in {f}. Skipping.")
                continue
            X = df[pkt_cols].values
            y = df['label'].values
            if len(X) == 0:
                continue
            all_X.append(X)
            all_y.append(y)
        except Exception as e:
            logger.warning(f"Failed to load {f}: {e}")

    if not all_X:
        logger.error("No valid data found.")
        sys.exit(1)

    X = np.vstack(all_X)
    y = np.hstack(all_y)

    logger.info(f"Class distribution: {collections.Counter(y)}")

    le = LabelEncoder()
    y = le.fit_transform(y)

    with open("label_classes.txt", "w") as f:
        for i, label in enumerate(le.classes_):
            f.write(f"{i}: {label}\n")

    imputer = SimpleImputer(strategy='mean')
    X = imputer.fit_transform(X)

    return X, y, le

# --- Training and Evaluation ---
def train_and_evaluate(X, y, le):
    logger.info(f"GPUs available: {torch.cuda.device_count()}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    writer = SummaryWriter(log_dir=f"runs/{RUN_NAME}")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold = 0
    all_metrics = []

    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y), y=y)
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

    for train_idx, val_idx in skf.split(X, y):
        fold += 1
        logger.info(f"Starting Fold {fold}")

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)

        if USE_PCA:
            pca = PCA(n_components=PCA_COMPONENTS)
            X_train = pca.fit_transform(X_train)
            X_val = pca.transform(X_val)

        X_train = torch.tensor(X_train, dtype=torch.float32)
        X_val = torch.tensor(X_val, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.long)
        y_val = torch.tensor(y_val, dtype=torch.long)

        train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=BATCH_SIZE)

        model = CNNLSTMModel(input_size=X_train.shape[1], num_classes=len(le.classes_))
        if torch.cuda.device_count() > 1:
            logger.info(f"Using {torch.cuda.device_count()} GPUs via DataParallel")
            model = nn.DataParallel(model)
        model = model.to(device)

        criterion = nn.CrossEntropyLoss(label_smoothing=0.1, weight=class_weights)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
        scaler = torch.cuda.amp.GradScaler()

        best_val_loss = float("inf")
        patience = EARLY_STOPPING_PATIENCE

        for epoch in range(EPOCHS):
            model.train()
            running_loss = 0.0
            progress = tqdm(train_loader, desc=f"Fold {fold} Epoch {epoch+1}/{EPOCHS}", leave=False)
            for xb, yb in progress:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                with torch.cuda.amp.autocast():
                    preds = model(xb)
                    loss = criterion(preds, yb)
                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                running_loss += loss.item()

            train_loss = running_loss / len(train_loader)

            model.eval()
            val_loss, correct, total = 0.0, 0, 0
            y_preds, y_true = [], []

            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    with torch.cuda.amp.autocast():
                        preds = model(xb)
                        loss = criterion(preds, yb)
                    val_loss += loss.item()
                    correct += (preds.argmax(dim=1) == yb).sum().item()
                    total += yb.size(0)
                    y_preds.extend(preds.argmax(dim=1).cpu().numpy())
                    y_true.extend(yb.cpu().numpy())

            val_loss /= len(val_loader)
            val_acc = correct / total
            scheduler.step()

            logger.info(f"Fold {fold} Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            writer.add_scalar(f"Fold{fold}/Train_Loss", train_loss, epoch)
            writer.add_scalar(f"Fold{fold}/Val_Loss", val_loss, epoch)
            writer.add_scalar(f"Fold{fold}/Val_Acc", val_acc, epoch)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), f"{MODEL_PATH.replace('.pt', f'_fold{fold}.pt')}")
                patience = EARLY_STOPPING_PATIENCE
            else:
                patience -= 1
                if patience == 0:
                    logger.info("Early stopping triggered.")
                    break

        cm = confusion_matrix(y_true, y_preds)
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_, ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title(f'Normalized Confusion Matrix - Fold {fold}')
        plt.savefig(f'confusion_matrix_fold{fold}.png')
        plt.close()

        report = classification_report(y_true, y_preds, target_names=le.classes_, output_dict=True)
        with open(f"classification_report_fold{fold}.json", "w") as f:
            json.dump(report, f, indent=2)

        all_metrics.append({"fold": fold, "val_loss": val_loss, "val_acc": val_acc})

    with open("summary_metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)

    writer.close()

if __name__ == "__main__":
    X, y, le = load_data()
    train_and_evaluate(X, y, le)