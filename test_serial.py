#!/usr/bin/env python
"""Test if pyserial is properly installed"""

try:
    import serial
    print(f"✓ serial module imported successfully")
    print(f"  Version: {serial.__version__}")
    print(f"  Location: {serial.__file__}")

    import serial.tools.list_ports
    print(f"✓ serial.tools.list_ports imported successfully")

    ports = serial.tools.list_ports.comports()
    print(f"\n✓ Found {len(ports)} serial ports:")
    for port in ports:
        print(f"  - {port.device}: {port.description}")

    print("\n✅ All imports successful! Ready to use CSI_MISO_app.py")

except ImportError as e:
    print(f"❌ Import error: {e}")
    print("\nTry running:")
    print("  pip install --upgrade pyserial")
