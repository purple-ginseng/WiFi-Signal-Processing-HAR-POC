import socket, time, csv, math

UDP_IP = "0.0.0.0"
UDP_PORT = 12345
CSV_FILE = "csi_log.csv"
NUM_SUBCARRIERS = 64

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))
print(f"Logging to {CSV_FILE}...")

with open(CSV_FILE, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["timestamp", "subcarrier_index", "I", "Q", "magnitude", "phase"])

    start = time.time()
    while True:
        data, addr = sock.recvfrom(4096)
        line = data.decode('utf-8', errors='ignore').strip()
        if not line.startswith("CSI:"): continue

        values = [int(v) for v in line[4:].split(',') if v.strip().lstrip('-').isdigit()]
        now = round(time.time() - start, 4)

        for i in range(0, len(values) - 1, 2):
            sc = i // 2
            if sc >= NUM_SUBCARRIERS: break
            I, Q = values[i], values[i+1]
            mag = math.sqrt(I**2 + Q**2)
            phase = math.degrees(math.atan2(Q, I))
            writer.writerow([now, sc, I, Q, round(mag, 3), round(phase, 2)])
