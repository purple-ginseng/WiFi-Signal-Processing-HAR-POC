
import re
from typing import List, Dict
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import subprocess
import os
import shutil
import tempfile


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

def parse_bfm_report(bfm_report: str):
    """
    Parses a verbose packet dissection to find the Epoch Time, MAC addresses,
    and the BFM report data using a resilient parsing strategy.

    It prioritizes finding MAC addresses within parentheses but falls back to
    finding the last MAC address on a line if parentheses are not present.

    Returns:
        dict: A dictionary containing the parsed data with keys:
              'timestamp', 'mac_addresses', and 'feedback_matrices'.
    """
    # Unchanged regex patterns for timestamp and BFM data
    timestamp_regex = re.compile(r"Epoch (?:Arrival )?Time:\s*(\d+\.\d+)")
    bfm_regex = re.compile(r"SCIDX:\s*(-?\d+),\s*φ11:\s*(\d+),\s*ψ21:\s*(\d+)")

    # --- UPDATED: A more robust MAC address regex ---
    # This pattern attempts to find a MAC in parentheses first. If that fails,
    # it falls back to finding the last valid MAC address at the end of the line.
    mac_regex = re.compile(
        r"^(?P<label>(?:Receiver|Destination|Transmitter|Source) address):.*?"
        r"(?:\((?P<mac_paren>(?:[a-fA-F0-9]{2}:){5}[a-fA-F0-9]{2})\)|(?P<mac_last>(?:[a-fA-F0-9]{2}:){5}[a-fA-F0-9]{2}))$"
    )

    # Initialize containers
    timestamp = None
    mac_addresses = {}
    parsed_feedback_matrices = []
    
    for line in bfm_report.splitlines():
        # --- 1. Search for the timestamp (no change) ---
        if not timestamp:
            ts_match = timestamp_regex.search(line)
            if ts_match:
                timestamp = ts_match.group(1)
        
        # --- 2. Search for MAC Addresses with the new, resilient regex ---
        mac_match = mac_regex.search(line.strip())
        if mac_match:
            key = mac_match.group('label').lower().replace(' ', '_')
            # Prioritize the parenthesized MAC, otherwise use the fallback.
            mac_address = mac_match.group('mac_paren') or mac_match.group('mac_last')
            mac_addresses[key] = mac_address

        # --- 3. Search for BFM data (no change) ---
        bfm_match = bfm_regex.search(line.strip())
        if bfm_match:
            parsed_feedback_matrices.append({
                "SCIDX": int(bfm_match.group(1)),
                "phi11": int(bfm_match.group(2)),
                "psi21": int(bfm_match.group(3))
            })
            
    return {
        "timestamp": timestamp,
        "mac_addresses": mac_addresses,
        "feedback_matrices": parsed_feedback_matrices
    }   


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

    def _readable_path(self, pcap_path):
        """Return a path tshark can actually open, staging a copy if it can't.

        Ubuntu 24.04+ ships an AppArmor profile (/etc/apparmor.d/tshark) that
        confines tshark to abstractions/user-tmp plus a few system paths — it
        grants no read access under @{HOME}. So `tshark -r ~/proj/x.pcap` dies
        with "You don't have permission to read the file" (exit 3) even though
        the file is owned by the caller and readable by every other tool. The
        capture pipeline sees that as zero extractable packets.

        Copying into the temp directory the profile *does* allow sidesteps it
        without touching system security policy. On an unconfined host the
        probe succeeds and nothing is copied.

        Returns (path_to_read, staging_dir_or_None); the caller removes the
        staging dir when done.
        """
        def can_read(path):
            return subprocess.run(
                [self.tshark_path, '-r', str(path), '-c', '1'],
                capture_output=True, text=True
            )

        probe = can_read(pcap_path)
        if probe.returncode == 0:
            return str(pcap_path), None

        staging_dir = tempfile.mkdtemp(prefix='bfm_tshark_')
        staged = os.path.join(staging_dir, os.path.basename(str(pcap_path)))
        try:
            shutil.copy2(str(pcap_path), staged)
        except OSError as e:
            shutil.rmtree(staging_dir, ignore_errors=True)
            print(f"⚠️ Could not stage {pcap_path} for tshark: {e}")
            return str(pcap_path), None

        if can_read(staged).returncode != 0:
            # Staging didn't help (e.g. TMPDIR also lives under $HOME, or the
            # file is genuinely unreadable). Fall back and let the normal path
            # surface tshark's own error.
            shutil.rmtree(staging_dir, ignore_errors=True)
            print(f"⚠️ tshark cannot read {pcap_path}: {probe.stderr.strip()}")
            return str(pcap_path), None

        print(f"[Extractor] tshark cannot read {pcap_path} directly "
              f"(AppArmor profile on tshark); using a staged copy.")
        return staged, staging_dir

    # Replace your pcap_to_csv method with this corrected version.
    def pcap_to_csv(self, pcap_path: str, csv_path: str):
        """
        Processes a pcap file, extracts BFM reports, and saves them to a CSV file.
        """
        read_path, staging_dir = self._readable_path(pcap_path)
        try:
            return self._pcap_to_csv(read_path, csv_path, source_path=pcap_path)
        finally:
            if staging_dir:
                shutil.rmtree(staging_dir, ignore_errors=True)

    def _pcap_to_csv(self, pcap_path: str, csv_path: str, source_path=None):
        display_filter = "wlan.fixed.category_code == 21"
        
        # (The packet counting section remains the same as the previous correct version)
        print(f"Pre-calculating packet count in {source_path or pcap_path}...")
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
        mac_address_list = []
        
        try:
            process = subprocess.Popen(command, stdout=subprocess.PIPE, text=True, encoding='utf-8')
            packet_stream = self._packet_stream_parser(process.stdout)

            for bfm_report in tqdm(packet_stream, total=total_packets, desc="Extracting BFM"):
                bfm_report = clean_from_ansi(bfm_report)
                
                # --- NEW: Unpack the tuple returned by the updated function ---
                
                parsed_data = parse_bfm_report(bfm_report)
                timestamp, parsed_bfm = parsed_data['timestamp'], parsed_data['feedback_matrices']
                mac_addresses = parsed_data['mac_addresses']


                # We need both a timestamp and BFM data to proceed
                if not parsed_bfm or not timestamp:
                    continue

                # --- NEW: Add the found timestamp to our list ---
                timestamps_list.append(timestamp)
                
                data_for_df.append(parsed_bfm_to_list(parsed_bfm))

                mac_address_list.append(mac_addresses)
                
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
        df = pd.concat((pd.DataFrame(mac_address_list), df), axis = 1)
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
    import glob
    # Dynamically find all PCAP files in bfm_pcap directory
    files = sorted(glob.glob('bfm_pcap/*.pcap'))
    print(f"Found {len(files)} PCAP files to extract")
    # Use the macOS tshark path or find it in PATH
    extractor = BFMExtractor(tshark_path= '/usr/local/bin/tshark', csv_dir= 'bfm_raw_csv')
    extractor.extract(files)
    print(extractor.get_extracted_files())