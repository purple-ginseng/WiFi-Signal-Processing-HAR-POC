import socket
import csv
import datetime

UDP_IP = "0.0.0.0"
UDP_PORT = 12345

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
output_file = f"esp32_csi_{timestamp}.csv"

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))

print(f"[INFO] Saving CSI to {output_file}")

try:
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        while True:
            data, addr = sock.recvfrom(4096)
            line = data.decode().strip()

            # Assume CSI data is CSV-formatted
            row = line.split(",")
            writer.writerow(row)

            print(f"[RECEIVED] {len(row)} values from {addr}")
except KeyboardInterrupt:
    print(f"\n[INFO] Done. Saved to {output_file}")
finally:
    sock.close()
