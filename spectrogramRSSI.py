import argparse
import os
import re
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
import torch
import torch.nn.functional as F

def sorted_pkt_cols(df):
    pkt_cols = [c for c in df.columns if re.fullmatch(r"pkt\d+", c)]
    return sorted(pkt_cols, key=lambda x: int(x[3:]))

def single_attention(seq_1d: torch.Tensor) -> np.ndarray:
    x = seq_1d.unsqueeze(-1)                 # (T, 1)
    scores = torch.matmul(x, x.T) / seq_1d.shape[0]**0.5
    attn = F.softmax(scores, dim=-1)
    return attn.cpu().numpy()               # (T, T)

def save_spectrogram(signal, title, out_path, fs=60):
    f, t, Sxx = spectrogram(signal, fs=fs, nperseg=16, noverlap=8)
    plt.figure(figsize=(10, 3))
    plt.pcolormesh(t, f, Sxx, shading="gouraud")
    plt.ylabel("Freq [Hz]")
    plt.xlabel("Time [s]")
    plt.title(title)
    plt.colorbar(label="Power")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

def save_attention(attn, title, out_img_path, out_npy_path):
    plt.figure(figsize=(4, 4))
    plt.imshow(attn, cmap="viridis")
    plt.title(title)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(out_img_path, dpi=300)
    plt.close()
    
    np.save(out_npy_path, attn)

def process_csv(csv_path, out_dir, fs):
    df = pd.read_csv(csv_path)
    pkt_cols = sorted_pkt_cols(df)
    if len(pkt_cols) != 60:
        print(f"[{os.path.basename(csv_path)}] skipped (expected 60 pkt columns). Found {len(pkt_cols)}.")
        return

    base = os.path.splitext(os.path.basename(csv_path))[0]
    for idx, row in df.iterrows():
        series = row[pkt_cols].astype(float).values
        label  = str(row.get("label", "NA"))
        signal_name = f"{base}_row{idx:03d}_{label}"

        # 1. Spectrogram
        spec_path = os.path.join(out_dir, f"{signal_name}_spec.png")
        save_spectrogram(series, f"Spectrogram – {signal_name}", spec_path, fs)

        # 2. Attention matrix
        rssi_tensor = torch.tensor(series, dtype=torch.float32)
        attn = single_attention(rssi_tensor)
        attn_img_path = os.path.join(out_dir, f"{signal_name}_attn.png")
        attn_npy_path = os.path.join(out_dir, f"{signal_name}_attn.npy")
        save_attention(attn, f"Attention – {signal_name}", attn_img_path, attn_npy_path)

    print(f"✓ {os.path.basename(csv_path)} → processed {len(df)} rows")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="./data", help="Folder with wifisignal_data_*.csv")
    ap.add_argument("--out_dir", default="./output", help="Where to save outputs")
    ap.add_argument("--fs", type=float, default=60.0, help="Sampling frequency for spectrogram")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    files = glob.glob(os.path.join(args.data_dir, "wifisignal_data_*.csv"))
    if not files:
        print("❌ No CSV files matching pattern found in the data directory.")
        return

    for f in files:
        process_csv(f, args.out_dir, args.fs)

if __name__ == "__main__":
    main()
