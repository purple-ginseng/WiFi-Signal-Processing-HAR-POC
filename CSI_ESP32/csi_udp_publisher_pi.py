import serial
import socket
import time
import subprocess
import os

# === CONFIGURATION ===
SERIAL_PORT = '/dev/ttyUSB0'
BAUD_RATE = 115200
RECONNECT_DELAY = 5
MAX_BUFFER_SIZE = 65536
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
            ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.05)
            print(f"[INFO] Connected to {SERIAL_PORT}")
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
print(f"[INFO] UDP target: {UDP_TARGET_IP}:{UDP_TARGET_PORT}")

buffer = ""
ser = None

try:
    ser = connect_serial()
    
    while True:
        try:
            if ser.in_waiting:
                raw_bytes = ser.read(ser.in_waiting)
                if raw_bytes:
                    buffer += raw_bytes.decode('utf-8', errors='ignore')
                    
                    # Prevent buffer overflow
                    if len(buffer) > MAX_BUFFER_SIZE:
                        print("[WARN] Buffer overflow, clearing...")
                        buffer = buffer[-MAX_BUFFER_SIZE//2:]

                    # Process complete lines
                    lines = buffer.split('\n')
                    buffer = lines[-1]  # save last (possibly incomplete) part
                    
                    for line in lines[:-1]:
                        line = line.strip()
                        if line.startswith("CSI:"):
                            try:
                                # Extract CSI data after "CSI:" prefix
                                csi_data = line[4:].strip()  # Remove "CSI:" prefix
                                
                                # main_gui.py expects format: "I1,Q1,I2,Q2,I3,Q3,..."
                                # where each value is an integer (positive or negative)
                                # Validate format before sending
                                if csi_data and ',' in csi_data:
                                    # Basic validation: check if it contains comma-separated data
                                    values = csi_data.split(',')
                                    if len(values) >= 2:  # At least one I/Q pair
                                        payload = csi_data.encode('utf-8')
                                        sock.sendto(payload, (UDP_TARGET_IP, UDP_TARGET_PORT))
                                        print("[SENT]", csi_data[:50] + "..." if len(csi_data) > 50 else csi_data)
                                    else:
                                        print("[WARN] CSI data too short, skipping:", csi_data[:30])
                                else:
                                    print("[WARN] Invalid CSI format, skipping:", line[:30])
                            except socket.error as e:
                                print(f"[ERROR] UDP send failed: {e}")

            time.sleep(0.001)
            
        except serial.SerialException as e:
            print(f"[ERROR] Serial connection lost: {e}")
            if ser:
                ser.close()
            print("[INFO] Attempting to reconnect...")
            ser = connect_serial()
            buffer = ""  # Clear buffer on reconnection

except KeyboardInterrupt:
    print("\n[INFO] Stopped by user.")
finally:
    if ser and ser.is_open:
        ser.close()
    sock.close()