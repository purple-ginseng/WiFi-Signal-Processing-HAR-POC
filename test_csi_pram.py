import serial, time, statistics, re, sys

# ========= USER CONFIG =========
PORT = "/dev/cu.usbmodem5A7C1148511"   # Receiver ESP32 port
BAUD = 115200
STATS_INTERVAL = 2.0                   # seconds between stats updates
EXPECTED_RATE_MIN = 20                 # minimum expected packets/sec (warning threshold)
# ===============================

ser = serial.Serial(PORT, BAUD, timeout=1)
print(f"🔗 Connected to {PORT}")
print(f"📊 Monitoring CSI packet rate (target: 20-100 Hz for HAR)\n")

rssi_values = []
pkt_count = 0
total_packets = 0
start = time.time()
session_start = time.time()

pattern = re.compile(r"rssi=(-?\d+)")

try:
    while True:
        line = ser.readline().decode(errors='ignore').strip()
        if not line:
            continue

        # Count CSI packets (look for "CSI" prefix)
        if line.startswith("CSI"):
            pkt_count += 1
            total_packets += 1

            # Extract RSSI
            m = pattern.search(line)
            if m:
                rssi = int(m.group(1))
                rssi_values.append(rssi)
                if len(rssi_values) > 200:
                    rssi_values.pop(0)

        # Display stats every STATS_INTERVAL seconds
        elapsed = time.time() - start
        if elapsed >= STATS_INTERVAL:
            avg_rssi = round(statistics.mean(rssi_values), 1) if rssi_values else 0
            rate = round(pkt_count / elapsed, 1)
            session_time = int(time.time() - session_start)

            # Status indicator based on rate
            if rate < EXPECTED_RATE_MIN:
                status = "⚠️  LOW"
            elif rate < 40:
                status = "✓ OK"
            else:
                status = "✓✓ GOOD"

            print(f"{status} | Packets: {pkt_count:5d} | Rate: {rate:6.1f}/s | "
                  f"Avg RSSI: {avg_rssi:5.1f} dBm | Total: {total_packets:6d} | "
                  f"Time: {session_time}s")

            sys.stdout.flush()
            start = time.time()
            pkt_count = 0

except KeyboardInterrupt:
    print("\n\n✋ Stopped by user.")
    session_duration = time.time() - session_start
    avg_rate = total_packets / session_duration if session_duration > 0 else 0
    print(f"📈 Session Summary:")
    print(f"   Total packets: {total_packets}")
    print(f"   Duration: {session_duration:.1f}s")
    print(f"   Average rate: {avg_rate:.1f} packets/s")
finally:
    ser.close()
