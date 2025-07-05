
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
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

# Device selection
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Configuration
DATA_DIR = "./data"
PCA_COMPONENTS = 35
TEST_SIZE = 0.4
EPOCHS = 50
BATCH_SIZE = 16
USE_PCA = True

# Load RSSI Data
def load_rssi_data():
    files = glob.glob(os.path.join(DATA_DIR, "wifisignal_data_*.csv"))
    data, labels = [], []
    for f in files:
        df = pd.read_csv(f)
        pkt_cols = [c for c in df.columns if c.startswith("pkt")]
        X = df[pkt_cols].values.astype(float)
        y = df["label"].values
        data.append(X)
        labels.append(y)
    X = np.vstack(data)
    y = np.hstack(labels)
    return X, y

# CNN-LSTM Model
class CNNLSTM(nn.Module):
    def __init__(self, input_size, num_classes):
        super(CNNLSTM, self).__init__()
        self.conv1 = nn.Conv1d(1, 64, kernel_size=3)
        self.pool = nn.MaxPool1d(2)
        self.lstm = nn.LSTM(input_size=(input_size - 2)//2, hidden_size=64, batch_first=True)
        self.fc1 = nn.Linear(64, 64)
        self.fc2 = nn.Linear(64, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = x.unsqueeze(1)  # (B, 1, input_size)
        x = self.pool(F.relu(self.conv1(x)))  # (B, 64, L)
        x = x.permute(0, 2, 1)  # (B, L, 64)
        _, (h_n, _) = self.lstm(x)
        x = self.dropout(F.relu(self.fc1(h_n[-1])))
        return self.fc2(x)

def train_model():
    X, y = load_rssi_data()
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    num_classes = len(le.classes_)

    X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=TEST_SIZE, stratify=y_enc)
    scaler = StandardScaler()

    if USE_PCA:
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        pca = PCA(n_components=PCA_COMPONENTS)
        X_train = pca.fit_transform(X_train_scaled)
        X_test = pca.transform(X_test_scaled)
        input_shape = PCA_COMPONENTS
    else:
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        input_shape = X.shape[1]

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test_tensor, y_test_tensor), batch_size=BATCH_SIZE)

    model = CNNLSTM(input_shape, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            output = model(xb)
            loss = criterion(output, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {running_loss/len(train_loader):.4f}")

    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            output = model(xb)
            preds = torch.argmax(output, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(yb.numpy())

    cm = confusion_matrix(all_labels, all_preds)
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=le.classes_, yticklabels=le.classes_)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig("confusion_matrix_pytorch.png")

    report = classification_report(all_labels, all_preds, target_names=le.classes_)
    print(report)
    with open("classification_report_pytorch.txt", "w") as f:
        f.write(report)

if __name__ == "__main__":
    train_model()
