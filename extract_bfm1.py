import pyshark
import re
import pandas as pd
from typing import List, Dict

# Set file path and display filter
filepath = "./data/beamforming_capture.pcap"
display_filter = (
    "wlan.fixed.category_code == 21 and wlan.vht.action == 0 and "
    "wlan.vht.mimo_control.ncindex == 0x000001 and wlan.vht.mimo_control.nrindex == 0x000001"
)

# Initialize capture
cap = pyshark.FileCapture(input_file=filepath, display_filter=display_filter)

def clean_from_ansi(s: str) -> str:
    ansi_escape_pattern = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    return ansi_escape_pattern.sub('', s)

def parse_bfm(bfm_report: str) -> List[Dict]:
    regex_pattern = re.compile(r"SCIDX:\s*(-?\d+),\s*φ11:\s*(\d+),\s*ψ21:\s*(\d+)")
    parsed_feedback_matrices = []
    for line in bfm_report.splitlines():
        match = regex_pattern.search(line.strip())
        if match:
            scidx = int(match.group(1))
            phi11 = int(match.group(2))
            psi21 = int(match.group(3))
            parsed_feedback_matrices.append({
                "SCIDX": scidx,
                "phi11": phi11,
                "psi21": psi21
            })
    return parsed_feedback_matrices

def parsed_bfm_to_list(bfm: List[Dict]) -> List[int]:
    data = []
    for subcarrier in bfm:
        data.append(subcarrier['phi11'])
        data.append(subcarrier['psi21'])
    return data

def parsed_bfm_list_to_df(bfm_list: List[List[Dict]]) -> pd.DataFrame:
    # Generate column names based on first packet
    columns = []
    for subcarrier in bfm_list[0]:
        scidx = subcarrier['SCIDX']
        columns.append(f"SCIDX_{scidx}_phi11")
        columns.append(f"SCIDX_{scidx}_psi21")
    
    data = [parsed_bfm_to_list(bfm) for bfm in bfm_list]
    return pd.DataFrame(data=data, columns=columns)

# Process packets
bfm_list = []

for packet in cap:
    try:
        bfm_report = str(packet['wlan.mgt'])
        bfm_report = clean_from_ansi(bfm_report)
        parsed_bfm = parse_bfm(bfm_report)
        if parsed_bfm:  # Only append if non-empty
            bfm_list.append(parsed_bfm)
    except Exception as e:
        print(f"Error processing packet: {e}")

# Convert to DataFrame and save to CSV
if bfm_list:
    df = parsed_bfm_list_to_df(bfm_list)
    output_csv = "bfm_extracted.csv"
    df.to_csv(output_csv, index=False)
    print(f"Saved extracted BFM data to {output_csv}")
else:
    print("No valid BFM data extracted.")
