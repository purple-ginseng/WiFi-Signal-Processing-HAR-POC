# WiFi Signal Processing for Human Activity Recognition

Device-free human activity recognition (HAR) using IEEE 802.11ac/ax **Compressed Beamforming (BFM)** reports captured from a commodity OpenWrt router. The system records BFM angles (`phi`, `psi`) emitted by associated WiFi clients, decompresses them into per-subcarrier complex channel ratios, and feeds the resulting time-frequency representation to a CNN for activity classification (e.g. *Standing* vs *Walking*).

A live Streamlit page can pipe data straight from the router over SSH/SFTP and run inference in real time, including a self-ping mode that turns the router into both transmitter and receiver for radar-like sensing.

## Pipeline overview

![Pipeline overview](plots/fig_pipeline_overview.png)

| Stage | Tool | Output |
|-------|------|--------|
| 1. RF capture | `tcpdump` on `mon0` (router) | `*.pcap` |
| 2. Frame parsing | `bfmtool/extractor.py` (calls `tshark`) | per-packet BFM angles CSV |
| 3. Decompression | `bfmtool/preprocessor.py` | per-subcarrier complex ratio (Real/Imag) |
| 4. Feature extraction | `convert_real_imag_to_mag_phase` / `compute_doppler_features` | magnitude, phase, Doppler velocity |
| 5. Classification | Keras CNN (`best_bfm_model_*.keras`) | activity label + probability |

## Hardware requirements

- **Router**: OpenWrt 24.10 device with a 5 GHz radio that can be put into monitor mode. Reference platform used in this repo: **D-Link DIR-842 C2** (`ath79`, `mips_24kc`).
- **Capture interface**: `mon0` on `phy0`, brought up as `type monitor`.
- **Workstation**: macOS, Linux, or Windows with Python 3.10+ and Wireshark/`tshark` installed.
- **Connection**: Ethernet from the workstation to the router's LAN port. The router does not need to be online; it only needs to reach the workstation over the LAN cable.

## Software requirements

### On the workstation

- Python 3.10+ (3.11–3.13 tested)
- Wireshark / `tshark`
  - macOS: `/Applications/Wireshark.app/Contents/MacOS/tshark`
  - Linux: `/usr/bin/tshark`
  - Windows: `C:\Program Files\Wireshark\tshark.exe`
- Python deps: `pip install -r requirements.txt` plus `paramiko`, `streamlit`, `tensorflow` for the live page.

### On the router (one-time setup)

The OpenWrt base image does **not** ship `tcpdump`, and the default SSH server (`dropbear`) does **not** expose an SFTP subsystem. Both have to be added once. The instructions below assume the router is at `192.168.1.1` with `root:123456` (the defaults the code uses).

1. Determine the router's package architecture:
   ```bash
   ssh root@192.168.1.1 'opkg print-architecture; cat /etc/openwrt_release'
   ```
2. From a workstation that *does* have internet, download three packages (substitute the architecture you saw above; example below is `mips_24kc` on OpenWrt 24.10.0):
   ```bash
   BASE=https://downloads.openwrt.org/releases/24.10.0/packages/mips_24kc
   curl -O $BASE/base/libpcap1_1.10.5-r2_mips_24kc.ipk
   curl -O $BASE/base/tcpdump-mini_4.99.5-r1_mips_24kc.ipk
   curl -O $BASE/packages/openssh-sftp-server_9.9_p2-r1_mips_24kc.ipk
   ```
3. Copy them onto the router and install:
   ```bash
   scp -O *.ipk root@192.168.1.1:/tmp/
   ssh root@192.168.1.1 'opkg install /tmp/libpcap1_*.ipk \
                                     /tmp/tcpdump-mini_*.ipk \
                                     /tmp/openssh-sftp-server_*.ipk'
   ```
4. Sanity-check the capture path (no traffic generated yet — empty pcap is expected):
   ```bash
   ssh root@192.168.1.1 'tcpdump -i mon0 -c 1 -w /tmp/test.pcap "wlan[24] == 21"; ls -lh /tmp/test.pcap'
   ```

If `mon0` is missing on first boot, the collector code in `bfmtool/collector.py` will recreate it automatically (`iw phy phy0 interface add mon0 type monitor`); you can do the same by hand.

## Repository layout

```
.
|-- BFMapp.py                       # Streamlit explorer for offline BFM CSVs
|-- main_gui.py                     # Tk GUI for labeled data collection
|-- pages/
|   |-- live_activity_detection.py  # Live Streamlit prediction page (multi-page entry)
|   `-- realtime_activity_test.py
|-- bfmtool/                        # Core capture + processing library
|   |-- collector.py                # SSH/paramiko + tcpdump on mon0
|   |-- extractor.py                # tshark-driven pcap -> CSV of BFM angles
|   |-- preprocessor.py             # Givens-rotation decompression to complex ratios
|   |-- data_loader.py
|   |-- model.py
|   `-- utils.py
|-- train_bfm_by_environment.ipynb  # CNN trainer per environment (open / nofoil / foil)
|-- train_bfm_doppler.ipynb         # Doppler-feature trainer
|-- train_csi_model.py              # Legacy CSI baseline
|-- best_bfm_model_*.keras          # Trained CNN checkpoints
|-- bfm_*_pca.pkl                   # Fitted PCA transformers
|-- bfm_pcap/, bfm_raw_csv/, bfm_processed_csv/        # Labeled-collection outputs
|-- live_bfm_pcap/, live_bfm_raw_csv/, live_bfm_processed_csv/  # Live-mode outputs
|-- plots/                          # Pipeline + per-step example figures
`-- ConfusionMatrixes/              # Per-fold evaluation figures
```

## Quick start: live activity detection

End-to-end smoke test, assuming the router is prepped per the section above and the cable is plugged in.

1. Install Python deps and start Streamlit:
   ```bash
   pip install -r requirements.txt paramiko streamlit tensorflow
   streamlit run pages/live_activity_detection.py
   ```
2. Open <http://localhost:8501>.
3. Expand **Router Diagnostics & Setup** and click **Test Connection via BFMCollector** to confirm SSH + SFTP work.
4. Click **Start Live Detection**. The collector will:
   - Open paramiko SSH to `192.168.1.1`
   - Start `iperf3 -s` and `ping -i 0.01 192.168.1.1` on the router (self-ping at ~100 Hz to keep BFM frames flowing — see [Self-ping / radar-like mode](#self-ping--radar-like-mode))
   - Run `tcpdump -i mon0 -p -w /tmp/bfm_capture -W 10 -C 1 'wlan[24] == 21'`
   - Pull rotated pcaps via SFTP, decompress, and predict
5. Watch **Status** flip from red to green and **Total Packets** grow. After ~5 packets a prediction (Standing / Walking) appears with class probabilities and live magnitude / phase / Doppler plots.

To stop, click **Stop Detection** — this kills the remote `tcpdump`, `iperf3`, and `ping` cleanly.

### Self-ping / radar-like mode

`ENABLE_TRAFFIC_GENERATION` in `pages/live_activity_detection.py` (line 59) toggles the traffic source:

- `True` (default): the router pings itself at 100 Hz and runs `iperf3 -s`. The same device acts as transmitter and receiver, giving radar-style monostatic-ish sensing without needing an external client.
- `False`: relies on natural traffic from a phone or laptop already associated with the AP. Useful when you want passive bistatic measurements.

## Data collection workflow (labeled)

For training data, use the Tk GUI rather than the Streamlit page:

```bash
python main_gui.py
```

1. Select source mode **BFM-PCAP**, then click **Setup BFM**. This SSHs into the router and starts `iperf3 -s`.
2. Pick an activity label from the dropdown and a duration (seconds).
3. Click **Collect**. Pcaps stream into `bfm_pcap/<label>_<timestamp>N.pcap`.
4. After collection, the GUI automatically:
   - Calls `BFMExtractor.extract` -> `bfm_raw_csv/<...>.csv` (BFM angles per packet)
   - Calls `BFMPreprocessor.process` -> `bfm_processed_csv/<...>.csv` (complex per-subcarrier ratios)

Repeat per (subject, environment, activity) combination.

## Signal processing, step by step

Each plot shows the same packet window after a successive transformation. Magnitude is on the left column, phase on the right.

| Step | Description | Magnitude | Phase |
|------|-------------|-----------|-------|
| 1 | Raw decompressed complex BFM ratio | ![](plots/step1.png) | (combined view) |
| 2 | Per-subcarrier mag/phase split | ![](plots/step2mag.png) | ![](plots/step2phase.png) |
| 3 | Reference-subtracted (DC removal) | ![](plots/step3mag.png) | ![](plots/step3phase.png) |
| 4 | Phase unwrap + linear-trend removal | ![](plots/step4mag.png) | ![](plots/step4phase.png) |
| 5 | Hampel outlier filtering | ![](plots/step5mag.png) | ![](plots/step5phase.png) |
| 6 | Smoothing (uniform / Gaussian) | ![](plots/step6mag.png) | ![](plots/step6phase.png) |
| 7 | Detrend | ![](plots/step7mag.png) | ![](plots/step7phase.png) |
| 8 | Final feature representation fed to the model | ![](plots/step8mag.png) | ![](plots/step8phase.png) |

Inspect or replay these stages interactively with the offline explorer:

```bash
streamlit run BFMapp.py
```

Recent additions to `BFMapp.py`: temporal phase traces and PCA(n=1) traces over magnitude and phase with explained-variance annotation.

## Training

Two reference notebooks:

- `train_bfm_by_environment.ipynb` — splits by environment (`open`, `nofoil`, `foil`) and trains a per-environment CNN. Produces `best_bfm_model_<env>.keras`, `bfm_<env>_pca.pkl`, and confusion matrices.
- `train_bfm_doppler.ipynb` — uses Doppler-velocity features instead of raw magnitude/phase. Produces `best_bfm_doppler_<env>.keras`.

Both notebooks expect the labeled `bfm_processed_csv/` directory described above.

### Representative results

PCA explained variance over BFM features (used for dimensionality reduction before the dense head):

![PCA explained variance](pca_explained_variance.png)

Per-environment training history and confusion matrices:

| | Magnitude/phase model | Doppler model |
|---|---|---|
| Training history | ![](bfm_all_environments_history.png) | ![](bfm_doppler_training_history.png) |
| Confusion matrix | ![](bfm_all_environments_confusion.png) | ![](bfm_doppler_confusion_matrices.png) |

Cross-environment accuracy comparison:

![Accuracy comparison](bfm_environment_accuracy_comparison.png)

Phase representation sanity check (raw vs circular-mean):

![Circular mean phase](circular_mean_phase_comparison.png)

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `tcpdump: not found` on router | Package missing on stock OpenWrt | Install `libpcap1` + `tcpdump-mini` (see [On the router](#on-the-router-one-time-setup)) |
| Streamlit log shows `[Download] SFTP error: EOF during negotiation` repeating | Dropbear has no SFTP subsystem | Install `openssh-sftp-server` on the router |
| `tcpdump failed to start after interface reset` | `mon0` is missing or in use | `iw dev mon0 del; iw phy phy0 interface add mon0 type monitor; ip link set mon0 up` |
| `WARNING: REMOTE HOST IDENTIFICATION HAS CHANGED!` after reflash | Stale host key | `ssh-keygen -R 192.168.1.1` then SSH again to accept the new key |
| `Could not resolve host: downloads.openwrt.org` while preparing the router | macOS default route went out the LAN cable to the OpenWrt (no WAN there) | Ensure Wi-Fi is up and is the primary service in System Settings -> Network, or `sudo route change default <wifi-gateway>` temporarily |
| Status stays at "Connecting..." | SSH credentials wrong, or router unreachable | Verify with `ssh root@192.168.1.1`; check `ifconfig en9` (or your USB-Ethernet name) is up |
| `Total Packets` stays at 0 with tcpdump running | No client traffic on the AP -> no BFM frames | Enable self-ping mode (`ENABLE_TRAFFIC_GENERATION = True`) or associate a phone to the AP and use it |
| Predictions stuck on one class | Model trained on a different environment | Switch model in the sidebar (`nofoil` vs `open`) or retrain with new data |

## SOP: from cold start to first prediction

1. Power on router. Confirm the AP is broadcasting on 5 GHz (`iw dev` shows `phy0-ap0`).
2. Plug LAN cable from workstation into a router LAN port.
3. `ssh root@192.168.1.1` and confirm `tcpdump --version` and `/usr/libexec/sftp-server -h` both succeed. If not, run [On the router](#on-the-router-one-time-setup).
4. On the workstation: `streamlit run pages/live_activity_detection.py`.
5. In the browser, open the Diagnostics expander and click **Test Connection via BFMCollector**. Expect a green check.
6. Click **Start Live Detection**. Within ~5 s, **Status** turns green and **Total Packets** climbs.
7. Stand still for 10 s, then walk back and forth for 10 s. Watch the predicted class flip in the Current Prediction panel.
8. Click **Stop Detection** before unplugging the cable to leave the router clean (kills `tcpdump`, `iperf3`, `ping`).

## Citation

If you use this code in academic work, please cite the paper draft in `BFM/paper.md`.
