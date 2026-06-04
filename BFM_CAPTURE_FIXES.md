# BFM Capture Stability Fixes

Branch: `fixes/bfm-capture-stability`  
File: `main_gui2.py`

---

## 1. tcpdump filter broadened and `-p` flag removed

**Problem:** The original filter `wlan[24] == 21` only targeted 802.11ac (VHT) BFM
frames. Routers running 802.11ax (WiFi 6) send category-30 HE BFM frames instead,
which were silently missed. The `-p` (no-promiscuous) flag was also present, which is
semantically wrong on a monitor interface and suppresses frames on some drivers.

**Fix:**
```
# Before
tcpdump -i mon0 -p -U -B 4096 -G 1 -W 10 -w /tmp/bfm_capture 'wlan[24] == 21'

# After
tcpdump -i mon0 -U -B 4096 -G 1 -W 10 -w /tmp/bfm_capture 'wlan[24] == 21 or wlan[24] == 30'
```

- `-p` removed — monitor interfaces must be in promiscuous mode to see all frames.
- Filter extended to catch both VHT (cat 21) and HE/WiFi-6 (cat 30) action frames.
- The filter is now stored as `_tcpdump_cmd` and can be overridden by the progressive
  preflight test (see §4).

---

## 2. Stall-detection threshold raised from 0.5 s to 5.0 s

**Problem:** `STALL_THRESHOLD = 0.5` was shorter than tcpdump's own rotation period
(`-G 1` = 1 second). Every normal rotation where no frame arrived in the last 0.5 s
was misidentified as a stall, causing tcpdump to be killed and restarted in a tight
loop — clearing `/tmp/bfm_capture*` and losing any partially-written file.

**Fix:**
```python
self.STALL_THRESHOLD = 5.0  # was 0.5
```

The threshold is now well above the 1-second rotation window, so the watchdog only
fires after a genuine multi-second silence.

---

## 3. Persistent SFTP session (replaces open/close at 10 Hz)

**Problem:** Every 100 ms poll iteration opened a new SFTP channel with
`client.open_sftp()` and closed it with `sftp.close()`. At 10 calls/second this
exhausted the SSH channel multiplexer and caused sporadic `Channel … is not open`
errors, dropping file downloads mid-session.

**Fix:**  
One SFTP session is opened before the download loop begins and reused across all
iterations. It is only reopened if an exception occurs (session lost / SSH reconnect):

```python
sftp = None

def _open_sftp():
    nonlocal sftp
    if sftp is not None:
        try: sftp.close()
        except Exception: pass
    sftp = self.bfm_collector.client.open_sftp()

_open_sftp()   # open once

while self.running:
    try:
        if sftp is None:
            _open_sftp()
        remote_files = sftp.listdir("/tmp/")
        ...
    except Exception as e:
        sftp.close(); sftp = None   # reopen next iteration
```

---

## 4. Progressive preflight filter test (3-pass, 3 s each)

**Problem:** On a fresh router the correct tcpdump filter was unknown. If the channel
was wrong or the router used a non-standard PHY, no BFM frames would arrive but there
was no feedback explaining why.

**Fix:**  
`_preflight_check` now runs three 3-second test captures in descending specificity:

| Pass | Filter expression | Description |
|------|-------------------|-------------|
| 1 | `wlan[24] == 21 or wlan[24] == 30` | VHT + HE BFM candidates |
| 2 | `(wlan[0] & 0xfc) == 0xd0` | Any Action frame |
| 3 | *(no filter)* | All 802.11 frames |

The first pass that produces a pcap > 24 bytes (i.e., at least one frame) wins.
Its filter expression is stored in `self._bfm_tcpdump_filter` and passed to
`LiveDataCollector` so the live capture uses exactly the proven filter.

If all three passes return 0 packets, `mon0` is on the wrong channel and the user
is told to enter the channel manually.

---

## 5. mon0 channel locking fixed for 5 GHz

**Problem:** Auto-detection only looked for `wlan0`/`wlan1`/`wlan0-1` (the Linux
default names), missing OpenWrt interfaces like `phy0-ap0` or `ath0`. Even when a
channel was found, `iw dev mon0 set channel N` does not set the channel width on
5 GHz, so mon0 silently stayed on a 20 MHz slice of a different center frequency.

**Fix:**  
1. `iw dev` is parsed fully — every interface block is iterated; the first non-`mon0`
   interface with a `channel` line is used as the AP's current operating channel.
2. `iw dev mon0 set freq <MHz> <width>` is used instead of `set channel`:
   ```bash
   iw dev mon0 set freq 5180 80MHz   # for 80 MHz wide 5 GHz channel
   ```
3. A manual override field ("mon0 channel") lets the user specify the channel
   directly if auto-detection fails.

---

## 6. Ping target corrected from router IP to WiFi client IP

**Problem:** The initial ping targeted `192.168.1.1` (the router itself). The router
handles this via its kernel loopback — the ping never goes over the radio. BFM frames
are triggered by traffic exchanged between the AP and an associated WiFi client, so
a loopback ping generates zero BFM frames.

**Fix:**  
- The UI has a configurable "Ping target IP (WiFi client)" field (default `192.168.1.2`).
- `_restart_router_ping()` reads this field and sends the ping to the actual client.
- The initial connection ping in `_connect_with_retry` also reads the UI value via a
  `_app_ping_target_ip` reference set on the collector instance.

---

## 7. tcpdump health-check restart method fixed

**Problem:** The download loop's 10-second health check called
`self.bfm_collector.run_tcpdump()` — the `BFMCollector` library's built-in method.
This method uses different flags (no `-G`, no `-W`, different filter) from the
explicit command used at startup, producing an inconsistent capture configuration
after a restart.

**Fix:**  
Health check now calls `self._launch_tcpdump_explicit(verify=False)`, which re-issues
exactly the same command stored in `self._tcpdump_cmd`:

```python
# Before
self.bfm_collector.run_tcpdump()

# After
self._launch_tcpdump_explicit(verify=False)
```

---

## 8. BFM yield tracking and >100% yield bug fixed

**Problem:** Two separate tshark invocations were used — one to count raw frames,
one to extract BFM data. If the first invocation failed silently (tshark exits
non-zero; `subprocess.run` does not raise by default), `total_raw_frames` was
incremented by 0. When the extraction then succeeded, `total_packets / 0` or
`total_packets / near-zero` produced yields well above 100%.

**Fix:**  
A single combined tshark pass reads both `frame.number` and
`wlan.fixed.category_code` at once:

```python
r = subprocess.run(
    [tshark_path, "-r", pcap_path, "-T", "fields",
     "-e", "frame.number", "-e", "wlan.fixed.category_code"],
    capture_output=True, text=True, encoding="utf-8",
)
for line in r.stdout.splitlines():
    parts = line.split("\t")
    if not parts[0].strip():   # skip empty lines, not count("\n")
        continue
    raw_frame_count += 1
    cat = parts[1].strip() if len(parts) > 1 else ""
    if cat == "21":  cat21_count += 1
    elif cat == "30": cat30_count += 1
```

`total_packets` is also now counted from the CSV row count (the ground truth of
successfully parsed BFM frames) rather than `len(df_mag_phase)`.

The yield metric is now displayed in the GUI:
```
✓ BFM yield: 12/15 (80.0%) — cat21=12, cat30=0, other=3
```
A yield consistently below 10% indicates a channel mismatch or non-BFM traffic.

---

## 9. Cross-session packet contamination fixed

**Problem:** The download/processing pipeline runs continuously between labeled
sessions (so preflight doesn't need to re-run). At the start of a new session,
three contamination paths existed:

| Path | Effect |
|------|--------|
| `packet_buffer` not cleared | Session 2's CSV included all packets ever captured |
| `download_queue` not cleared | Pcap files queued at the tail of session 1 were processed into session 2 |
| No timestamp guard | A pcap straddling the session boundary injected old rows |

**Fix (3 layers):**

1. **Queue clearing** at session start:
   ```python
   self.bfm_collector.download_queue.clear()
   self.bfm_collector.packet_buffer.clear()
   self.bfm_collector.processed_buffer.clear()
   ```

2. **Timestamp filter on in-RAM buffer**: `session_start_ts = time.time()` is
   recorded before clearing. At save time, any row with
   `timestamp < session_start_ts` is dropped:
   ```python
   ts_numeric = pd.to_numeric(df_out["timestamp"], errors="coerce")
   df_out = df_out[ts_numeric >= session_start_ts]
   ```

3. **Timestamp filter in snapshot**: `_snapshot_session_to_bfm_dirs` receives
   `min_timestamp` and filters every merged CSV the same way.
