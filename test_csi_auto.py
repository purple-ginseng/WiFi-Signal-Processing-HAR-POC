import serial, time, re, csv, math, statistics, sys
from datetime import datetime

# ========= USER CONFIG =========
PORT = "/dev/cu.usbmodem5A7C1148511"   # adjust if needed
BAUD = 921600                          # Match ESP32 baud rate (increased for high-speed capture)
ACTIVITY_NAME = "testing"              # Activity name for filename (e.g., "walking", "sitting", "testing")
OUTPUT_DIR = "./data"                  # Directory to save CSV files
WINDOW_DURATION = 90                   # seconds per CSV file
PRINT_STATS = True
STATS_INTERVAL = 2.0                   # seconds between stats updates
EXPECTED_RATE_MIN = 5                  # minimum expected packets/sec (warning threshold)
# ===============================

# Auto-detect CSI format:
# Format 1: CSI ts=12345 rssi=-85 len=128 [mac=...] (followed by data line)
# Format 2: CSI:value,value,value,...,RSSI:value (all in one line)
meta_pattern = re.compile(r"ts=(\d+)\s+rssi=(-?\d+)\s+len=(\d+)")
mac_pattern = re.compile(r"mac=([0-9A-F:]+)")
simple_csi_pattern = re.compile(r"^CSI:([\d,\-]+)")
rssi_pattern = re.compile(r"RSSI:(-?\d+)")

# Create output directory if it doesn't exist
import os
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    print(f"📁 Created directory: {OUTPUT_DIR}")

def generate_filename(activity, start_time):
    """Generate filename: esp32_csi_{activity}_{yyyymmdd}_{timestamp}.csv"""
    dt = datetime.fromtimestamp(start_time)
    date_str = dt.strftime("%Y%m%d")
    timestamp_str = dt.strftime("%H%M%S")
    filename = f"esp32_csi_{activity}_{date_str}_{timestamp_str}.csv"
    return os.path.join(OUTPUT_DIR, filename)

def create_new_csv(filename):
    """Create a new CSV file with headers"""
    csv_file = open(filename, mode="w", newline="")
    writer = csv.writer(csv_file)
    writer.writerow(["timestamp", "subcarrier_index", "I", "Q", "magnitude", "phase_deg", "rssi", "mac_address"])
    return csv_file, writer

# Open serial
ser = serial.Serial(PORT, BAUD, timeout=1)
print(f"🔗 Connected to {PORT} @ {BAUD} baud")
print(f"📊 Activity: {ACTIVITY_NAME}")
print(f"⏱️  Window duration: {WINDOW_DURATION}s per file")
print(f"📂 Output directory: {OUTPUT_DIR}\n")

# Initialize first CSV file
session_start = time.time()
window_start = time.time()
current_filename = generate_filename(ACTIVITY_NAME, window_start)
csv_file, writer = create_new_csv(current_filename)
print(f"📝 Recording to: {os.path.basename(current_filename)}")

# Stats tracking
start = time.time()
pkt_count, total_packets, window_packets = 0, 0, 0
file_count = 1
rssi_values = []
mac_addresses = {}

# Track startup phase
startup_phase = True
startup_lines = []
startup_timeout = time.time() + 10

try:
    while True:
        # Check if we need to rotate to a new CSV file
        window_elapsed = time.time() - window_start
        if window_elapsed >= WINDOW_DURATION:
            # Close current file
            csv_file.close()
            window_duration = time.time() - window_start
            print(f"\n✅ File complete: {os.path.basename(current_filename)}")
            print(f"   Duration: {window_duration:.1f}s | Packets: {window_packets}\n")

            # Create new file
            window_start = time.time()
            current_filename = generate_filename(ACTIVITY_NAME, window_start)
            csv_file, writer = create_new_csv(current_filename)
            file_count += 1
            window_packets = 0
            print(f"📝 Recording to: {os.path.basename(current_filename)} (File #{file_count})")

        line = ser.readline().decode(errors="ignore").strip()
        if not line:
            # Check if we've been waiting too long for startup
            if startup_phase and time.time() > startup_timeout:
                print("⚠️  Startup messages not detected - ESP32 may have already started")
                print("   Starting CSI packet monitoring...\n")
                startup_phase = False
            continue

        # Capture startup messages (scan results, etc.)
        if startup_phase:
            startup_lines.append(line)
            # Check if we've finished startup
            if "Listening for CSI packets" in line or "🎧" in line or line.startswith("CSI"):
                startup_phase = False
                if len(startup_lines) > 5:
                    print("\n" + "="*60)
                    print("STARTUP SCAN RESULTS:")
                    print("="*60)
                    for sline in startup_lines:
                        if any(x in sline for x in ["🔍", "📊", "Rank", "PRIMARY", "Target", "✅", "⚠️"]):
                            print(sline)
                    print("="*60 + "\n")
                else:
                    print("⚠️  ESP32 already running - missed startup scan")
                    print("   (Press RESET on ESP32 to see scan results)\n")
                print("Now monitoring CSI packet capture:\n")
                if not line.startswith("CSI"):
                    continue

        # Parse CSI lines - supports two formats
        if line.startswith("CSI"):
            # Check for simple format: CSI:value,value,value,...,RSSI:value
            simple_match = simple_csi_pattern.match(line)

            if simple_match:
                # Simple format (ESP32 promiscuous mode)
                ts = int(time.time() * 1e6)  # microseconds
                rssi = 0
                mac_addr = "PROMISCUOUS"

                # Extract RSSI if present in the line
                rssi_match = rssi_pattern.search(line)
                if rssi_match:
                    rssi = int(rssi_match.group(1))

                # Extract CSI data from the same line (everything before RSSI)
                csi_data_str = simple_match.group(1)
                # Remove RSSI part if it's included in the CSI data string
                if "RSSI:" in csi_data_str:
                    csi_data_str = csi_data_str.split(",RSSI:")[0]

                try:
                    raw_values = [int(x) for x in csi_data_str.split(",") if x.strip() != ""]
                except ValueError:
                    continue

            else:
                # Metadata format: CSI ts=... rssi=... len=...
                m = meta_pattern.search(line)
                if not m:
                    continue

                ts = int(m.group(1))
                rssi = int(m.group(2))
                csi_len = int(m.group(3))

                # Extract MAC address if present
                mac_match = mac_pattern.search(line)
                mac_addr = mac_match.group(1) if mac_match else "UNKNOWN"

                # Next line(s) contain the CSI payload
                raw_line = ser.readline().decode(errors="ignore").strip()
                if not raw_line:
                    continue

                try:
                    raw_values = [int(x) for x in raw_line.split(",") if x.strip() != ""]
                except ValueError:
                    continue

            # Track MAC addresses
            if mac_addr not in mac_addresses:
                mac_addresses[mac_addr] = 0
            mac_addresses[mac_addr] += 1

            pkt_count += 1
            total_packets += 1
            window_packets += 1
            rssi_values.append(rssi)
            if len(rssi_values) > 200:
                rssi_values.pop(0)

            # CSI packets are pairs of (I,Q)
            num_subcarriers = len(raw_values) // 2
            for i in range(num_subcarriers):
                I = raw_values[2*i]
                Q = raw_values[2*i + 1]
                mag = math.sqrt(I**2 + Q**2)
                phase = math.degrees(math.atan2(Q, I))
                writer.writerow([ts/1e6, i, I, Q, round(mag,3), round(phase,3), rssi, mac_addr])

        # Display stats every STATS_INTERVAL seconds
        elapsed = time.time() - start
        if PRINT_STATS and elapsed >= STATS_INTERVAL:
            avg_rssi = round(statistics.mean(rssi_values), 1) if rssi_values else 0
            rate = round(pkt_count / elapsed, 1)
            session_time = int(time.time() - session_start)
            window_time = int(time.time() - window_start)
            window_remaining = max(0, WINDOW_DURATION - window_time)

            # Status indicator based on rate
            if rate < EXPECTED_RATE_MIN:
                status = "⚠️  LOW"
            elif rate < 20:
                status = "✓ OK"
            else:
                status = "✓✓ GOOD"

            # Build MAC summary
            mac_summary = ""
            if mac_addresses:
                primary_mac = max(mac_addresses, key=mac_addresses.get)
                mac_count = len(mac_addresses)
                if mac_count == 1:
                    mac_summary = f"MAC: {primary_mac}"
                else:
                    mac_summary = f"MACs: {mac_count} ({primary_mac[:8]}... primary)"

            print(f"{status} | Rate: {rate:6.1f}/s | Window: {window_time}s/{WINDOW_DURATION}s (−{window_remaining}s) | "
                  f"Pkts: {window_packets:5d} | Total: {total_packets:6d} | File #{file_count} | {mac_summary}")

            sys.stdout.flush()
            start = time.time()
            pkt_count = 0

except KeyboardInterrupt:
    print("\n\n✋ Stopped by user.")

    # Close current file
    csv_file.close()

    session_duration = time.time() - session_start
    avg_rate = total_packets / session_duration if session_duration > 0 else 0

    print(f"\n📈 Session Summary:")
    print(f"   Total packets: {total_packets}")
    print(f"   Duration: {session_duration:.1f}s")
    print(f"   Average rate: {avg_rate:.1f} packets/s")
    print(f"   Files created: {file_count}")

    if mac_addresses:
        print(f"\n📡 MAC Addresses Captured:")
        sorted_macs = sorted(mac_addresses.items(), key=lambda x: x[1], reverse=True)
        for mac, count in sorted_macs:
            percentage = 100.0 * count / total_packets if total_packets > 0 else 0
            print(f"   {mac}: {count:5d} packets ({percentage:5.1f}%)")

finally:
    ser.close()
    if not csv_file.closed:
        csv_file.close()
    print(f"\n💾 Last file saved to: {current_filename}")
    print(f"📂 All files in: {OUTPUT_DIR}")
