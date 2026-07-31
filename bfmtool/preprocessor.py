import numpy as np
import re
import pandas as pd
from pathlib import Path
from typing import List, Optional, Tuple

# 802.11 compressed-beamforming codebooks, as (phi_bit, psi_bit) pairs ordered
# from coarsest to finest. The angles are quantized with a codebook selected by
# the beamformer; decoding with the wrong pair rescales every angle and silently
# destroys dynamic range (a 6-bit phi read as 9-bit is compressed 8x).
#
#   HT  (802.11n):  Nb=2 -> (4, 2),  Nb=4 -> (6, 4)
#   VHT (802.11ac): Codebook Information 0 -> (7, 5),  1 -> (9, 7)
CODEBOOKS: Tuple[Tuple[int, int], ...] = ((4, 2), (6, 4), (7, 5), (9, 7))

# Used when the codebook cannot be inferred (e.g. an empty capture). Matches the
# value this module hardcoded before the codebook became configurable.
DEFAULT_PHI_BIT, DEFAULT_PSI_BIT = 9, 7


def infer_codebook(phi_values, psi_values, verbose: bool = True):
    """
    Infer (phi_bit, psi_bit) from the observed quantized angle indices.

    tshark reports phi11/psi21 as raw quantized integers, so the field width is
    recoverable from their range: a b-bit field takes values 0..2**b-1. We pick
    the coarsest standard codebook that can represent every observed index.

    This is a lower bound, not proof. If a capture never exercises the top of
    its range the inference will under-estimate, so the result is reported as
    'saturated' (some index hits exactly 2**b-1, which pins the width) or
    'unsaturated' (consistent with this codebook and any finer one). Prefer
    passing phi_bit/psi_bit explicitly when the link configuration is known.

    Returns:
        (phi_bit, psi_bit, saturated)
    """
    max_phi = np.nanmax(phi_values)
    max_psi = np.nanmax(psi_values)

    if not np.isfinite(max_phi) or not np.isfinite(max_psi):
        if verbose:
            print("[BFM] No usable angle indices; falling back to "
                  f"phi_bit={DEFAULT_PHI_BIT}, psi_bit={DEFAULT_PSI_BIT}.")
        return DEFAULT_PHI_BIT, DEFAULT_PSI_BIT, False

    max_phi, max_psi = int(max_phi), int(max_psi)

    for phi_bit, psi_bit in CODEBOOKS:
        if max_phi < 2 ** phi_bit and max_psi < 2 ** psi_bit:
            saturated = (max_phi == 2 ** phi_bit - 1) or (max_psi == 2 ** psi_bit - 1)
            if verbose:
                confidence = "saturated" if saturated else "UNSATURATED (lower bound)"
                print(f"[BFM] Inferred codebook phi_bit={phi_bit}, psi_bit={psi_bit} "
                      f"from index ranges phi<={max_phi}, psi<={max_psi} [{confidence}]")
                if not saturated:
                    print("[BFM] WARNING: neither angle reached the top of its range, so a "
                          "finer codebook cannot be ruled out. Pass phi_bit/psi_bit "
                          "explicitly if you know the link configuration.")
            return phi_bit, psi_bit, saturated

    raise ValueError(
        f"Observed angle indices (phi<={max_phi}, psi<={max_psi}) exceed every known "
        f"802.11 codebook {CODEBOOKS}. The extractor may be misparsing the BFM report."
    )


def decompress_bfm_from_angles(bfm_angles, Nc, Nr,
                               phi_bit: int = DEFAULT_PHI_BIT,
                               psi_bit: int = DEFAULT_PSI_BIT):
    """
    Decompresses BFM angles into a complex matrix using vectorized NumPy operations.

    phi_bit/psi_bit select the quantization codebook and MUST match the one the
    beamformer used; see CODEBOOKS and infer_codebook().
    """
    if Nr != 2 or Nc != 1:
        raise ValueError(f"This decompression function is for a 2x1 system, got Nr={Nr}, Nc={Nc}")

    # Bit precision constants. phi_bit/psi_bit may be scalars, or 1-D arrays of
    # length num_rows when a capture mixes links that use different codebooks —
    # reshaped to (num_rows, 1) so they broadcast across subcarriers.
    phi_bit = np.asarray(phi_bit, dtype=float)
    psi_bit = np.asarray(psi_bit, dtype=float)
    if phi_bit.ndim == 1:
        phi_bit = phi_bit[:, None]
    if psi_bit.ndim == 1:
        psi_bit = psi_bit[:, None]

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

LINK_COLS = ['transmitter_address', 'receiver_address']


def infer_codebook_per_link(df, phi_values, psi_values, locked: Optional[dict] = None,
                            verbose: bool = True):
    """
    Infer the codebook separately for each (transmitter, receiver) link.

    The codebook is a property of the beamformer, and a monitor-mode capture
    routinely picks up several. Inferring over a whole file lets a handful of
    stray packets from an unrelated AP pin the codebook for everyone — which
    silently mis-decodes the link you actually care about, since the coarsest
    consistent codebook is chosen from the pooled maximum.

    `locked` is an optional {link_key: (phi_bit, psi_bit)} carried across chunks
    of one session; it is updated in place with any newly inferred links.

    Returns per-row arrays (phi_bit, psi_bit) of length len(df).
    """
    n = len(df)
    phi_bits = np.empty(n, dtype=float)
    psi_bits = np.empty(n, dtype=float)

    if not all(c in df.columns for c in LINK_COLS):
        pb, sb, _ = infer_codebook(phi_values, psi_values, verbose=verbose)
        phi_bits[:] = pb
        psi_bits[:] = sb
        return phi_bits, psi_bits

    # .indices gives positional row numbers per group, which is what we need to
    # index the phi/psi arrays regardless of the frame's index.
    positions = df.groupby(LINK_COLS, sort=False).indices

    for key, rows in positions.items():
        if locked is not None and key in locked:
            pb, sb = locked[key]
        else:
            pb, sb, saturated = infer_codebook(phi_values[rows], psi_values[rows],
                                               verbose=False)
            if locked is not None:
                locked[key] = (pb, sb)
            if verbose:
                tx, rx = key
                flag = "" if saturated else "  [UNSATURATED - lower bound]"
                print(f"[BFM]   link {tx} -> {rx}: {len(rows):>5} pkts, "
                      f"phi_bit={pb}, psi_bit={sb}{flag}")
        phi_bits[rows] = pb
        psi_bits[rows] = sb

    if verbose and len(positions) > 1:
        print(f"[BFM] Capture mixes {len(positions)} links; each decoded with its own "
              f"codebook. Use filter_by_mode() downstream to keep only the target link.")

    return phi_bits, psi_bits


def process_bfm_dataframe(df, phi_bit: Optional[int] = None, psi_bit: Optional[int] = None,
                          locked_codebooks: Optional[dict] = None):
    """
    Loads and processes a BFM DataFrame using fully vectorized operations.

    phi_bit/psi_bit select the quantization codebook. Leave both as None to infer
    it per (transmitter, receiver) link from the observed angle indices; pass both
    to force a known codebook for every row.
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
    if phi_bit is None or psi_bit is None:
        phi_bit, psi_bit = infer_codebook_per_link(df, phi_values, psi_values,
                                                   locked=locked_codebooks)

    # The result 'bfm_complex' will have shape (num_rows, num_subcarriers, 2)
    bfm_complex = decompress_bfm_from_angles(bfm_angles, Nc=1, Nr=2,
                                             phi_bit=phi_bit, psi_bit=psi_bit)
    
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
    """
    Decodes raw BFM angle CSVs into complex antenna-ratio CSVs.

    The quantization codebook may be given explicitly (phi_bit/psi_bit) or left
    to be inferred per (transmitter, receiver) link. Inferred codebooks are
    LOCKED per link for the lifetime of the instance: a capture session is split
    into many one-second pcap chunks, and a short chunk may not exercise the full
    angle range, so re-inferring per chunk would rescale different parts of the
    same session differently. The first chunk that sees a link fixes its scale.
    """

    def __init__(self, dir : str,
                 phi_bit: Optional[int] = None,
                 psi_bit: Optional[int] = None):
        self.dir = Path(dir)
        self.processed_file = set()

        # Explicit => used for every row, never inferred.
        self.phi_bit = phi_bit
        self.psi_bit = psi_bit

        # {(transmitter, receiver): (phi_bit, psi_bit)} accumulated across chunks.
        self.locked_codebooks = {}

    def process_file(self, filepath : str, save_to : str):
        df = pd.read_csv(filepath)
        df = process_bfm_dataframe(df, phi_bit = self.phi_bit, psi_bit = self.psi_bit,
                                   locked_codebooks = self.locked_codebooks)
        df.to_csv(save_to, index = False)
        self.processed_file.add(save_to)

    def process(self, paths : List[str]):
        self.dir.mkdir(exist_ok = True)
        for file in paths:
            file = Path(file)
            self.process_file(file, self.dir / file.name)



if __name__ == '__main__':
    import glob
    preprocessor = BFMPreprocessor(dir = 'bfm_processed_csv')
    # Dynamically find all CSV files in bfm_raw_csv directory
    files = sorted(glob.glob('bfm_raw_csv/*.csv'))
    print(f"Found {len(files)} CSV files to process")
    preprocessor.process(files)
    

    


