import socket
import time

UDP_IP = "192.168.3.76"  # <-- Replace with your MacBook IP
UDP_PORT = 12345

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

print(f"[INFO] Sending UDP packets to {UDP_IP}:{UDP_PORT}...")

try:
    while True:
        message = "Hello from Raspberry Pi!"
        sock.sendto(message.encode(), (UDP_IP, UDP_PORT))
        print(f"[SENT] {message}")
        time.sleep(2)
except KeyboardInterrupt:
    print("\n[INFO] Sender stopped")
finally:
    sock.close()
