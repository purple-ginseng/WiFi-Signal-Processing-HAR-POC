import numpy as np
import re
import pandas as pd
from pathlib import Path
from typing import List

def decompress_bfm_from_angles(bfm_angles, Nc, Nr):
    """
    Decompresses BFM angles into a complex matrix using vectorized NumPy operations.
    """
    if Nr != 2 or Nc != 1:
        raise ValueError(f"This decompression function is for a 2x1 system, got Nr={Nr}, Nc={Nc}")

    # Bit precision constants
    phi_bit, psi_bit = 9, 7
    const1_phi, const2_phi = 1 / (2 ** (phi_bit - 1)), 1 / (2 ** phi_bit)
    const1_psi, const2_psi = 1 / (2 ** (psi_bit + 1)), 1 / (2 ** (psi_bit + 2))

    # --- Vectorized angle conversion ---
    # bfm_angles has shape (num_rows, num_subcarriers, 2)
    # Perform calculations on the entire arrays at once.
    phi_11 = np.pi * (const2_phi + const1_phi * bfm_angles[:, :, 0])
    psi_21 = np.pi * (const2_psi + const1_psi * bfm_angles[:, :, 1])

    # --- Vectorized Givens Rotation ---
    # Instead of a loop, we compute the components for all rows and subcarriers simultaneously.
    cos_psi = np.cos(psi_21)
    sin_psi = np.sin(psi_21)
    exp_phi = np.exp(1j * phi_11)
    
    # Calculate the two components of the V matrix directly
    # These are now 2D arrays of shape (num_rows, num_subcarriers)
    h1 = exp_phi * cos_psi
    h2 = sin_psi

    # Stack the results to get a complex matrix of shape (num_rows, num_subcarriers, 2)
    bfm_complex = np.stack([h1, h2], axis=-1)
    
    return bfm_complex

def process_bfm_dataframe(df):
    """
    Loads and processes a BFM DataFrame using fully vectorized operations.
    """
    print(f"Original data shape: {df.shape}")

    # --- 1. Vectorized Data Extraction ---
    # Extract subcarrier indices just once to build column names
    subcarrier_indices = sorted(list(set([int(re.search(r'SCIDX_(-?\d+)_', col).group(1)) for col in df.columns if col.startswith('SCIDX')])))
    num_subcarriers = len(subcarrier_indices)
    print(f"Found {num_subcarriers} subcarriers.")

    # Select columns for phi and psi in the correct order
    phi_cols = [f'SCIDX_{scidx}_phi11' for scidx in subcarrier_indices]
    psi_cols = [f'SCIDX_{scidx}_psi21' for scidx in subcarrier_indices]


    # Extract all data into NumPy arrays in one go. No loops!
    phi_values = df[phi_cols].values # Shape: (num_rows, num_subcarriers)
    psi_values = df[psi_cols].values # Shape: (num_rows, num_subcarriers)

    # Stack them into a single 3D array. Shape: (num_rows, num_subcarriers, 2)
    bfm_angles = np.stack([phi_values, psi_values], axis=2)

    # --- 2. Vectorized Decompression ---
    # The result 'bfm_complex' will have shape (num_rows, num_subcarriers, 2)
    bfm_complex = decompress_bfm_from_angles(bfm_angles, Nc=1, Nr=2)
    
    # --- 3. Vectorized BFM-Ratio Calculation ---
    h1 = bfm_complex[:, :, 0]  # Data from Antenna 1. Shape: (num_rows, num_subcarriers)
    h2 = bfm_complex[:, :, 1]  # Data from Antenna 2. Shape: (num_rows, num_subcarriers)

    # Avoid division by zero
    h2[h2 == 0] = 1e-9
    
    # The ratio is now calculated for all rows and subcarriers in a single operation
    bfm_ratio = h1 / h2 # Shape: (num_rows, num_subcarriers)

    # --- 4. Vectorized DataFrame Creation ---
    columns = []
    for scidx in subcarrier_indices:
        columns.append(f"SCIDX_{scidx}_Ratio_Real")
        columns.append(f"SCIDX_{scidx}_Ratio_Imag")
        
    # Interleave the real and imaginary parts efficiently
    final_data = np.empty((bfm_ratio.shape[0], bfm_ratio.shape[1] * 2))
    final_data[:, 0::2] = bfm_ratio.real
    final_data[:, 1::2] = bfm_ratio.imag
    
    ratio_df = pd.DataFrame(final_data, columns=columns)

    # add back columns
    other_cols = [col for col in df.columns if col not in phi_cols + psi_cols]
    ratio_df = pd.concat((df[other_cols], ratio_df), axis = 1)

    return ratio_df


class BFMPreprocessor():
    def __init__(self, dir : str):
        self.dir = Path(dir)
        self.processed_file = set()

    def process_file(self, filepath : str, save_to : str):
        df = pd.read_csv(filepath)
        df = process_bfm_dataframe(df)
        df.to_csv(save_to, index = False)
        self.processed_file.add(save_to)

    def process(self, paths : List[str]):
        self.dir.mkdir(exist_ok = True)
        for file in paths:
            file = Path(file)
            self.process_file(file, self.dir / file.name)



if __name__ == '__main__':
    preprocessor = BFMPreprocessor(dir = 'bfm_processed_csv')
    files = [r"D:\Matthew\SelfStudy\csi-project\PG_project\5 October 2025\WiFi-Signal-Processing-HAR\bfm_raw_csv\bfm_data_walking_ivan_alt_nofoil_20251009_184305.csv",
r"D:\Matthew\SelfStudy\csi-project\PG_project\5 October 2025\WiFi-Signal-Processing-HAR\bfm_raw_csv\bfm_data_walking_ivan_alt_nofoil_20251009_184644.csv",
r"D:\Matthew\SelfStudy\csi-project\PG_project\5 October 2025\WiFi-Signal-Processing-HAR\bfm_raw_csv\bfm_data_walking_kenny_alt_nofoil_20251009_181204.csv",
r"D:\Matthew\SelfStudy\csi-project\PG_project\5 October 2025\WiFi-Signal-Processing-HAR\bfm_raw_csv\bfm_data_walking_kenny_alt_nofoil_20251009_181639.csv",
r"D:\Matthew\SelfStudy\csi-project\PG_project\5 October 2025\WiFi-Signal-Processing-HAR\bfm_raw_csv\bfm_data_walking_kenny_alt_nofoil_20251009_182007.csv",
r"D:\Matthew\SelfStudy\csi-project\PG_project\5 October 2025\WiFi-Signal-Processing-HAR\bfm_raw_csv\bfm_data_walking_kenny_alt_nofoil_20251009_182338.csv",
r"D:\Matthew\SelfStudy\csi-project\PG_project\5 October 2025\WiFi-Signal-Processing-HAR\bfm_raw_csv\bfm_data_walking_kenny_alt_nofoil_20251009_182705.csv",
r"D:\Matthew\SelfStudy\csi-project\PG_project\5 October 2025\WiFi-Signal-Processing-HAR\bfm_raw_csv\bfm_data_standing_ivan_alt_nofoil_20251009_183050.csv",
r"D:\Matthew\SelfStudy\csi-project\PG_project\5 October 2025\WiFi-Signal-Processing-HAR\bfm_raw_csv\bfm_data_standing_ivan_alt_nofoil_20251009_183419.csv",
r"D:\Matthew\SelfStudy\csi-project\PG_project\5 October 2025\WiFi-Signal-Processing-HAR\bfm_raw_csv\bfm_data_standing_ivan_alt_nofoil_20251009_183754.csv",
r"D:\Matthew\SelfStudy\csi-project\PG_project\5 October 2025\WiFi-Signal-Processing-HAR\bfm_raw_csv\bfm_data_standing_ivan_alt_nofoil_20251009_184118.csv",
r"D:\Matthew\SelfStudy\csi-project\PG_project\5 October 2025\WiFi-Signal-Processing-HAR\bfm_raw_csv\bfm_data_standing_ivan_alt_nofoil_20251009_184449.csv",
r"D:\Matthew\SelfStudy\csi-project\PG_project\5 October 2025\WiFi-Signal-Processing-HAR\bfm_raw_csv\bfm_data_standing_kenny_alt_nofoil_20251009_180944.csv",
r"D:\Matthew\SelfStudy\csi-project\PG_project\5 October 2025\WiFi-Signal-Processing-HAR\bfm_raw_csv\bfm_data_standing_kenny_alt_nofoil_20251009_181349.csv",
r"D:\Matthew\SelfStudy\csi-project\PG_project\5 October 2025\WiFi-Signal-Processing-HAR\bfm_raw_csv\bfm_data_standing_kenny_alt_nofoil_20251009_181824.csv",
r"D:\Matthew\SelfStudy\csi-project\PG_project\5 October 2025\WiFi-Signal-Processing-HAR\bfm_raw_csv\bfm_data_standing_kenny_alt_nofoil_20251009_182155.csv",
r"D:\Matthew\SelfStudy\csi-project\PG_project\5 October 2025\WiFi-Signal-Processing-HAR\bfm_raw_csv\bfm_data_standing_kenny_alt_nofoil_20251009_182522.csv",
r"D:\Matthew\SelfStudy\csi-project\PG_project\5 October 2025\WiFi-Signal-Processing-HAR\bfm_raw_csv\bfm_data_walking_ivan_alt_nofoil_20251009_183235.csv",
r"D:\Matthew\SelfStudy\csi-project\PG_project\5 October 2025\WiFi-Signal-Processing-HAR\bfm_raw_csv\bfm_data_walking_ivan_alt_nofoil_20251009_183604.csv",
r"D:\Matthew\SelfStudy\csi-project\PG_project\5 October 2025\WiFi-Signal-Processing-HAR\bfm_raw_csv\bfm_data_walking_ivan_alt_nofoil_20251009_183936.csv"]
    preprocessor.process(files)
    

    


