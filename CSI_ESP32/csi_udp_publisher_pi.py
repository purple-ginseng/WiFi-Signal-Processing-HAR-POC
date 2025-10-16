import serial
import socket
import time
import subprocess
import os

# === CONFIGURATION ===
SERIAL_PORT = '/dev/ttyUSB0'
BAUD_RATE = 115200  # Increased to match ESP32
RECONNECT_DELAY = 5
MAX_BUFFER_SIZE = 131072  # Doubled buffer size
MAX_RECONNECT_ATTEMPTS = 3

UDP_TARGET_IP = "192.168.3.4"  # <- replace with your receiver IP
UDP_TARGET_PORT = 12345

def reset_usb_device():
    """Reset USB device by unbinding and rebinding the USB driver"""
    try:
        print("[INFO] Attempting to reset USB device...")
        
        # Find the USB device
        result = subprocess.run(['lsusb'], capture_output=True, text=True)
        if result.returncode != 0:
            print("[WARN] Could not list USB devices")
            return False
            
        # Look for ESP32 device (common VID:PID patterns)
        esp32_patterns = ['10c4:ea60', '1a86:7523', '0403:6001']
        device_found = False
        
        for pattern in esp32_patterns:
            if pattern in result.stdout:
                device_found = True
                break
                
        if not device_found:
            print("[WARN] ESP32 USB device not found in lsusb output")
            return False
            
        # Try to reset via usbreset if available
        try:
            subprocess.run(['usbreset', SERIAL_PORT], check=True, capture_output=True)
            print("[INFO] USB device reset successful")
            time.sleep(2)  # Wait for device to reinitialize
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass
            
        # Alternative: try to reset via driver unbind/bind
        try:
            # Find the USB device path
            device_path = None
            for root, _, files in os.walk('/sys/bus/usb/devices/'):
                if 'ttyUSB0' in str(files) or any(f'ttyUSB{i}' in str(files) for i in range(10)):
                    device_path = root
                    break
                    
            if device_path:
                driver_path = os.path.join(device_path, 'driver')
                if os.path.exists(driver_path):
                    # Unbind
                    with open(os.path.join(driver_path, 'unbind'), 'w') as f:
                        f.write(os.path.basename(device_path))
                    time.sleep(1)
                    # Rebind  
                    with open(os.path.join(driver_path, 'bind'), 'w') as f:
                        f.write(os.path.basename(device_path))
                    time.sleep(2)
                    print("[INFO] USB driver reset successful")
                    return True
        except (PermissionError, OSError) as e:
            print(f"[WARN] USB reset failed (may need sudo): {e}")
            
        return False
        
    except Exception as e:
        print(f"[ERROR] USB reset failed: {e}")
        return False

def connect_serial():
    """Attempt to connect to serial port with retry logic and USB reset"""
    attempt = 0
    while True:
        try:
            ser = serial.Serial(
                SERIAL_PORT,
                BAUD_RATE,
                timeout=0.01,  # Reduced timeout for faster polling
                write_timeout=0,
                inter_byte_timeout=None
            )
            ser.reset_input_buffer()  # Clear any stale data
            print(f"[INFO] Connected to {SERIAL_PORT} at {BAUD_RATE} baud")
            return ser
        except serial.SerialException as e:
            print(f"[ERROR] Failed to connect to {SERIAL_PORT}: {e}")
            attempt += 1

            # Try USB reset after failed attempts
            if attempt % MAX_RECONNECT_ATTEMPTS == 0:
                print(f"[INFO] {MAX_RECONNECT_ATTEMPTS} connection attempts failed, trying USB reset...")
                reset_usb_device()

            print(f"[INFO] Retrying in {RECONNECT_DELAY} seconds... (attempt {attempt})")
            time.sleep(RECONNECT_DELAY)

# === Setup UDP ===
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1048576)  # 1MB send buffer
print(f"[INFO] UDP target: {UDP_TARGET_IP}:{UDP_TARGET_PORT}")

buffer = bytearray()  # Use bytearray for better performance
ser = None
packet_count = 0
CSI_PREFIX = b"CSI:"

try:
    ser = connect_serial()

    while True:
        try:
            # Read all available data at once
            bytes_available = ser.in_waiting
            if bytes_available:
                raw_bytes = ser.read(bytes_available)
                buffer.extend(raw_bytes)

                # Prevent buffer overflow
                if len(buffer) > MAX_BUFFER_SIZE:
                    print(f"[WARN] Buffer overflow ({len(buffer)} bytes), clearing...")
                    buffer = buffer[-MAX_BUFFER_SIZE//2:]

                # Process complete lines
                while b'\n' in buffer:
                    line_end = buffer.index(b'\n')
                    line = buffer[:line_end]
                    buffer = buffer[line_end + 1:]

                    if line.startswith(CSI_PREFIX):
                        try:
                            # Extract CSI data after "CSI:" prefix
                            csi_data = line[4:].strip()

                            # Quick validation: check for comma-separated data
                            if b',' in csi_data and len(csi_data) > 10:
                                sock.sendto(csi_data, (UDP_TARGET_IP, UDP_TARGET_PORT))
                                packet_count += 1
                                if packet_count % 100 == 0:
                                    print(f"[SENT] {packet_count} packets")
                        except socket.error as e:
                            print(f"[ERROR] UDP send failed: {e}")

        except serial.SerialException as e:
            print(f"[ERROR] Serial connection lost: {e}")
            if ser:
                ser.close()
            print("[INFO] Attempting to reconnect...")
            ser = connect_serial()
            buffer.clear()  # Clear buffer on reconnection

except KeyboardInterrupt:
    print(f"\n[INFO] Stopped by user. Total packets sent: {packet_count}")
finally:
    if ser and ser.is_open:
        ser.close()
    sock.close()