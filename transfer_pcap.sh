#!/bin/bash
while true; do
    sshpass -p '123456' scp -O root@192.168.1.1:/tmp/csi.pcap ./data/wifisignal.pcap
    sleep 1
done