#!/bin/bash

ROUTER_IP="192.168.1.1"
REMOTE_PATH="/tmp/"
LOCAL_PATH="./data/"
BASE_FILENAME="wifi_capture" # Corresponds to -w wifi_capture.pcap
FILE_COUNT=5                # Corresponds to -W 5
COPY_INTERVAL=2             # How often to check for new files and copy (seconds)

mkdir -p "$LOCAL_PATH"

# Loop indefinitely to check for and copy the latest PCAP segment
while true; do
    # Determine the name of the currently active file on the router
    # This is a heuristic. tcpdump -C -W always rotates to the "next" file index.
    # E.g., if -W 5, it goes .pcap -> 1.pcap -> 2.pcap -> 3.pcap -> 4.pcap -> .pcap (overwrite)
    # We need to find the latest modified file.
    # This might be tricky with `ls -t` over ssh.

    # A more robust way: copy ALL files in the ring buffer, and the Python script
    # should process the combined data or only the latest files.
    # However, your Python script currently expects a single PCAP.

    # Let's simplify and just copy the *currently being written to* base file or latest numbered file
    # This might mean you miss some packets that went into older, rotated files.
    # For true full history, you'd need to parse all 5 files.
    # But for *live view* of recent data, copying the active file is often sufficient.

    # Let's try to copy the *latest modified* file from the ring buffer.
    # This is safer than guessing which one is active.
    LATEST_REMOTE_FILE=$(sshpass -p '123456' scp -O root@"$ROUTER_IP" "ls -t ${REMOTE_PATH}${BASE_FILENAME}*.pcap | head -n 1")

    if [ -z "$LATEST_REMOTE_FILE" ]; then
        echo "No PCAP files found on router yet. Waiting..."
        sleep "$COPY_INTERVAL"
        continue
    fi

    FILENAME_ONLY=$(basename "$LATEST_REMOTE_FILE")
    LOCAL_FILE="${LOCAL_PATH}${FILENAME_ONLY}"

    # Copy the latest file. This will overwrite the local copy.
    # It's okay if it's partially written, your Python script will handle it.
    echo "Copying $LATEST_REMOTE_FILE to $LOCAL_FILE"
    scp root@"$ROUTER_IP":"$LATEST_REMOTE_FILE" "$LOCAL_FILE"

    # Pause before checking again
    sleep "$COPY_INTERVAL"
done