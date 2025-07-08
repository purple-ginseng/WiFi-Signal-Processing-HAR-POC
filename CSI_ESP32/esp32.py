import serial
import socket
import time

# === CONFIGURATION ===
SERIAL_PORT = '/dev/ttyUSB0'
BAUD_RATE = 115200

UDP_TARGET_IP = "192.168.3.3"  # <- replace with your receiver IP
UDP_TARGET_PORT = 12345

# === Setup Serial + UDP ===
ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.1)
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

print(f"[INFO] Listening on {SERIAL_PORT} and sending to {UDP_TARGET_IP}:{UDP_TARGET_PORT}")

buffer = ""

try:
    while True:
        if ser.in_waiting:
            raw_bytes = ser.read(ser.in_waiting)
            buffer += raw_bytes.decode('utf-8', errors='ignore')

            # Process complete lines
            lines = buffer.split('\n')
            buffer = lines[-1]  # save last (possibly incomplete) part for next round
            for line in lines[:-1]:
                line = line.strip()
                if line.startswith("CSI:"):
                    payload = line.encode('utf-8')
                    sock.sendto(payload, (UDP_TARGET_IP, UDP_TARGET_PORT))
                    print("[SENT]", line)

        # Optional small sleep to prevent 100% CPU usage
        time.sleep(0.001)

except KeyboardInterrupt:
    print("\n[INFO] Stopped by user.")
    ser.close()
    sock.close()
