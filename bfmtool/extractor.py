
import re
from typing import List, Dict
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import subprocess
import os


def get_pcap_packet_count(pcap_path):
    """
    Gets the total number of packets in a pcap file using capinfos.

    This is the fastest and most memory-efficient method.

    Args:
        pcap_path (str): The path to the .pcap file.

    Returns:
        int: The number of packets in the file, or None if an error occurs.
    """
    try:
        # The '-c' flag tells capinfos to return only the packet count
        command = ['capinfos', '-c', pcap_path]
        
        # Run the command and capture the output
        result = subprocess.run(
            command, 
            capture_output=True, 
            text=True, 
            check=True # Raises an exception if the command fails
        )
        
        # The output is the number as a string, so we convert it to an integer
        return int(result.stdout.strip())
    
    except FileNotFoundError:
        print("Error: 'capinfos' command not found. Is Wireshark/tshark installed and in your PATH?")
        return None
    except subprocess.CalledProcessError as e:
        print(f"Error running capinfos: {e.stderr}")
        return None
    except ValueError:
        print("Error: Could not parse the output of capinfos.")
        return None
    

def clean_from_ansi(s : str):
    ansi_escape_pattern = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    s = ansi_escape_pattern.sub('', s)
    return s

def parse_bfm(bfm_report: str):
    """
    Parses a verbose packet dissection to find the Epoch Time and BFM report.
    Handles variations in the timestamp format.

    Returns:
        tuple: A tuple containing (timestamp_str, list_of_bfm_dicts).
               Returns (None, []) if data is not found.
    """
    # Regex for BFM lines (unchanged)
    bfm_regex = re.compile(r"SCIDX:\s*(-?\d+),\s*φ11:\s*(\d+),\s*ψ21:\s*(\d+)")
    
    # --- UPDATED: More flexible regex for the timestamp ---
    timestamp_regex = re.compile(r"Epoch (?:Arrival )?Time:\s*(\d+\.\d+)")
    
    parsed_feedback_matrices = []
    timestamp = None
    
    for line in bfm_report.splitlines():
        # Search for the timestamp on each line
        if not timestamp: # Only find the first timestamp
            ts_match = timestamp_regex.search(line)
            if ts_match:
                timestamp = ts_match.group(1)
        
        # Search for BFM data
        bfm_match = bfm_regex.search(line.strip())
        if bfm_match:
            scidx = int(bfm_match.group(1))
            phi11 = int(bfm_match.group(2))
            psi21 = int(bfm_match.group(3))
            parsed_feedback_matrices.append({
                "SCIDX": scidx,
                "phi11": phi11,
                "psi21": psi21
            })
            
    return timestamp, parsed_feedback_matrices

def parsed_bfm_to_list(bfm : List[Dict]):
    """
    convert the parsed bfm (for one packet) to list. 

    parameter
    bfm : [{'SCIDX': -122, 'phi11': 16, 'psi21': 10}, 
        {'SCIDX': -121, 'phi11': 16, 'psi21': 10},...]
        
    return
    data list : (number of subcarrier, )

    """
    assert isinstance(bfm, list)
    assert isinstance(bfm[0], dict)

    data = []
    for subcarrier in bfm:
        data.append(subcarrier['phi11'])
        data.append(subcarrier['psi21'])

    return data

def parsed_bfm_list_to_df(bfm_list : List[List[Dict]]) -> pd.DataFrame:
    """
    convert list of parsed bfm (for many packet) to dataframe

    parameter
    bfm_list : [bfm_1, bfm_2, ..., bfm_n] 
                where bfm_i = [{'SCIDX': -122, 'phi11': 16, 'psi21': 10}, 
                                {'SCIDX': -121, 'phi11': 16, 'psi21': 10},...]

    return
    dataframe with columns =
        [SCIDX_-122_phi11,  SCIDX_-122_psi21,  SCIDX_-121_phi11,  ...  ,SCIDX_121_psi21,  SCIDX_122_phi11,  SCIDX_122_psi21]
    """

    # generate column names
    columns = []
    for subcarrier in bfm_list[0]:
        scidx = subcarrier['SCIDX'] # subcarrier index
        columns.append(f"SCIDX_{scidx}_phi11")
        columns.append(f"SCIDX_{scidx}_psi21")

    data = []
    
    for bfm in bfm_list:
        data.append(parsed_bfm_to_list(bfm))
    
    return pd.DataFrame(data = data, columns = columns)
    



class BFMExtractor:
    """
    Extracts Beamforming (BFM) reports from pcap files into a CSV format
    by efficiently calling the tshark command-line tool.
    """
    def __init__(self, tshark_path: str, csv_dir : str):
        """
        Initializes the extractor.

        Args:
            tshark_path (str): The absolute path to the tshark executable.
                               Example: r"C:\Program Files\Wireshark\tshark.exe"
        """
        if not os.path.exists(tshark_path):
            raise FileNotFoundError(
                f"tshark executable not found at the specified path: {tshark_path}"
            )
        self.tshark_path = tshark_path
        self._extracted_files = set()
        self.csv_dir = Path(csv_dir)
    
    def get_extracted_files(self):
        return self._extracted_files
        
    def _packet_stream_parser(self, stdout_stream):
        """
        Parses the verbose output of 'tshark -V' and yields the text
        for one packet at a time.
        """
        packet_text = ""
        for line in stdout_stream:
            # Packets in 'tshark -V' output start with "Frame"
            if line.startswith('Frame '):
                if packet_text:
                    yield packet_text
                packet_text = line
            else:
                packet_text += line
        # Yield the last packet in the stream
        if packet_text:
            yield packet_text

    # Replace your pcap_to_csv method with this corrected version.
    def pcap_to_csv(self, pcap_path: str, csv_path: str):
        """
        Processes a pcap file, extracts BFM reports, and saves them to a CSV file.
        """
        display_filter = "wlan.fixed.category_code == 21"
        
        # (The packet counting section remains the same as the previous correct version)
        print(f"Pre-calculating packet count in {pcap_path}...")
        count_command = [self.tshark_path, '-r', pcap_path, '-Y', display_filter]
        total_packets = 0
        try:
            process = subprocess.Popen(count_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8')
            total_packets = sum(1 for _ in process.stdout)
            stderr_output = process.stderr.read()
            if stderr_output:
                print(f"Tshark counting error: {stderr_output.strip()}")
        except FileNotFoundError:
            print(f"Error: '{self.tshark_path}' not found.")
            return
        
        if total_packets == 0:
            print("No packets matching the filter were found. Aborting.")
            return
        print(f"Found {total_packets} packets to extract. Starting main extraction...")

        command = [self.tshark_path, '-r', pcap_path, '-Y', display_filter, '-V']
        
        data_for_df = []
        columns = []
        # --- NEW: A list to hold our timestamps ---
        timestamps_list = []
        
        try:
            process = subprocess.Popen(command, stdout=subprocess.PIPE, text=True, encoding='utf-8')
            packet_stream = self._packet_stream_parser(process.stdout)

            for bfm_report in tqdm(packet_stream, total=total_packets, desc="Extracting BFM"):
                bfm_report = clean_from_ansi(bfm_report)
                
                # --- NEW: Unpack the tuple returned by the updated function ---
                timestamp, parsed_bfm = parse_bfm(bfm_report)
                
                # We need both a timestamp and BFM data to proceed
                if not parsed_bfm or not timestamp:
                    continue

                # --- NEW: Add the found timestamp to our list ---
                timestamps_list.append(timestamp)
                
                data_for_df.append(parsed_bfm_to_list(parsed_bfm))
                
                if not columns:
                    for subcarrier in parsed_bfm:
                        scidx = subcarrier['SCIDX']
                        columns.append(f"SCIDX_{scidx}_phi11")
                        columns.append(f"SCIDX_{scidx}_psi21")

            process.wait()

        except Exception as e:
            print(f"An error occurred during the main extraction process: {e}")
            return

        if not data_for_df:
            print("Extraction finished, but no valid BFM reports were parsed.")
            return
            
        print(f"Creating DataFrame from {len(data_for_df)} packets...")
        df = pd.DataFrame(data=data_for_df, columns=columns)
        
        # --- NEW: Insert the timestamp column at the beginning of the DataFrame ---
        df.insert(0, 'timestamp', timestamps_list)
        
        df.to_csv(csv_path, index=False)
        self._extracted_files.add(csv_path)
        print(f"✅ BFM data saved to {csv_path}")

    def extract(self, files : List[Path]):
        self.csv_dir.mkdir(exist_ok= True)
        for file in files:
            file = Path(file)
            self.pcap_to_csv(file, self.csv_dir / (file.name[:-5] + '.csv'))

if __name__  == '__main__':
    files = [r'bfm_pcap\bfm0.pcap', r'bfm_pcap\bfm1.pcap']
    extractor = BFMExtractor(tshark_path= r"C:\Program Files\Wireshark\tshark.exe", csv_dir= 'bfm_csv')
    extractor.extract(files)
    print(extractor.get_extracted_files())