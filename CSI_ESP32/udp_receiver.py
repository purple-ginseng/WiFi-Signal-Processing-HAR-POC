import socket

UDP_IP = "0.0.0.0"      # Listen on all IPs
UDP_PORT = 12345

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))

print(f"[INFO] Listening for UDP packets on port {UDP_PORT}...")

try:
    while True:
        data, addr = sock.recvfrom(1024)
        print(f"[RECEIVED from {addr}] {data.decode().strip()}")
except KeyboardInterrupt:
    print("\n[INFO] Stopped by user")
finally:
    sock.close()
