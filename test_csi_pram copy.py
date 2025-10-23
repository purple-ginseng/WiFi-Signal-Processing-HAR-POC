import serial, time, re, csv, math, statistics, sys

# ========= USER CONFIG =========
PORT = "/dev/cu.usbmodem5A7C1148511"   # adjust if needed
BAUD = 115200
CSV_FILE = "csi_capture.csv"           # output filename
PRINT_STATS = True
# ===============================

# Regex to extract timestamp/rssi from the summary line
meta_pattern = re.compile(r"ts=(\d+)\s+rssi=(-?\d+)\s+len=(\d+)")

# Open serial and CSV file
ser = serial.Serial(PORT, BAUD, timeout=1)
print(f"Connected to {PORT}")

csv_file = open(CSV_FILE, mode="w", newline="")
writer = csv.writer(csv_file)
writer.writerow(["timestamp","subcarrier_index","I","Q","magnitude","phase_deg"])

start = time.time()
pkt_count, rssi_values = 0, []

try:
    while True:
        line = ser.readline().decode(errors="ignore").strip()
        if not line:
            continue

        # Parse CSI summary lines (metadata)
        if line.startswith("CSI"):
            m = meta_pattern.search(line)
            if not m:
                continue
            ts = int(m.group(1))
            rssi = int(m.group(2))
            pkt_count += 1
            rssi_values.append(rssi)
            if len(rssi_values) > 200:
                rssi_values.pop(0)

            # Next line(s) may contain the CSI payload (comma-separated)
            raw_line = ser.readline().decode(errors="ignore").strip()
            if not raw_line:
                continue

            # Convert CSV-style CSI values to list of ints
            try:
                raw_values = [int(x) for x in raw_line.split(",") if x.strip() != ""]
            except ValueError:
                continue

            # CSI packets are pairs of (I,Q)
            num_subcarriers = len(raw_values) // 2
            for i in range(num_subcarriers):
                I = raw_values[2*i]
                Q = raw_values[2*i + 1]
                mag = math.sqrt(I**2 + Q**2)
                phase = math.degrees(math.atan2(Q, I))
                # Convert timestamp to seconds (from microseconds)
                writer.writerow([ts/1e6, i, I, Q, round(mag,3), round(phase,3)])

        # Display quick stats every 2 seconds
        if PRINT_STATS and (time.time() - start > 2):
            avg_rssi = round(statistics.mean(rssi_values), 1) if rssi_values else 0
            rate = round(pkt_count / (time.time() - start), 1)
            print(f"Packets: {pkt_count:5d} | Rate: {rate:6.1f}/s | Avg RSSI: {avg_rssi:5.1f} dBm")
            pkt_count, start = 0, time.time()
            sys.stdout.flush()

except KeyboardInterrupt:
    print("\nStopped by user.")
finally:
    ser.close()
    csv_file.close()
    print(f"CSI data saved to {CSV_FILE}")
