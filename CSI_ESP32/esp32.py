import socket
import serial
import time

# --- CONFIGURATION ---
SERIAL_PORT = "/dev/cu.usbserial-xxxx"  # Change this to your actual ESP32 port
BAUD_RATE = 115200
UDP_IP = "192.168.1.100"                # IP of your Raspberry Pi
UDP_PORT = 12345

# --- SETUP ---
ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

print(f"[INFO] Reading from {SERIAL_PORT}, sending to {UDP_IP}:{UDP_PORT}")

try:
    while True:
        line = ser.readline().decode(errors='ignore').strip()
        if line.startswith("CSI:"):
            sock.sendto(line.encode(), (UDP_IP, UDP_PORT))
            print(f"[SENT] {line[:60]}...")
        time.sleep(0.005)
except KeyboardInterrupt:
    print("\n[INFO] Stopped.")
finally:
    ser.close()
    sock.close()
