---
name: bfm-architecture
description: Implementation-level architecture reference for this repo's active BFM (Beamforming Feedback Matrix) surface — the bfmtool/ capture pipeline, the internals of main_gui.py and main_gui3.py, BFMapp.py, and the BFM data-directory layout. Load before implementing or debugging inside bfmtool/, main_gui.py, main_gui3.py, or BFMapp.py.
---

# BFM architecture reference

This is implementation detail for the active BFM surface (`main_gui.py`, `main_gui3.py`, `BFMapp.py`, `bfmtool/`, `rtbfm.py`). See the project's `CLAUDE.md` for the higher-level framing (BFM-vs-legacy, active-surface list, the `rtbfm.window_features()` invariant) — this skill doesn't repeat it.

## Shared capture/processing library (`bfmtool/`)

Both `main_gui.py` and `main_gui3.py` drive the same router-capture pipeline through `bfmtool/`:

```
router (mon0, tcpdump)  -->  *.pcap
     |  bfmtool/collector.py   (BFMCollector: paramiko SSH — starts tcpdump/iperf3, syncs the router's clock,
     |                          pulls rotated pcaps over SFTP; host/user/pass default to 192.168.1.1/root/123456)
     v
bfmtool/extractor.py    (BFMExtractor: tshark-driven pcap -> per-packet BFM angle CSV, phi/psi per subcarrier)
     v
bfmtool/preprocessor.py (BFMPreprocessor.process -> decompress_bfm_from_angles: Givens-rotation decompression
     |                    of phi/psi into a complex per-subcarrier ratio, e.g. SCIDX_<n>_Ratio_Real/Imag)
     v
bfmtool/utils.py         (append_mag_phase / get_bfm_columns: complex ratio -> Ratio_Mag/Ratio_Phase columns;
                           used by both GUIs' local convert_real_imag_to_mag_phase() wrapper)
```

`TSHARK_PATH` is auto-detected per-OS near the top of both GUI files (Linux default `/usr/bin/tshark`).

## `main_gui.py` vs `main_gui3.py`

Both are Tk apps sharing the naming helpers (`_session_name`, `_bfm_csv_filename`, `convert_real_imag_to_mag_phase`) and the "Setup BFM" -> collect -> extract -> preprocess flow, but they differ in how they collect and what they predict with:

- **`main_gui.py`** collects a full BFM session before processing it: `_do_bfm_collection` runs `tcpdump` for the requested duration, then on stop extracts/preprocesses everything into `bfm_pcap/`, `bfm_raw_csv/`, `bfm_processed_csv/`, and merges the session into one `bfm_real_imag_csv/<...>.csv` + `bfm_mag_phase_csv/<...>.csv` pair (`_merge_bfm_session_csv`). This is the "BFM-PCAP" source mode in its Data Collection tab. The GUI also still carries older RSSI-PCAP/CSI-UDP collection modes and a matching CNN "Model Training"/"Real-time Prediction" pair (`best_model.keras`, `pca_tf.pkl`, `label_encoder_tf.pkl`, `scaler.pkl`, `./data/*.csv`) predating the BFM work — dead weight carried along in the same file, not something to extend.
- **`main_gui3.py`** is the actively-developed real-time path: `LiveDataCollector` streams pcap chunks continuously into `live_bfm_pcap/` -> `live_bfm_raw_csv/` -> `live_bfm_processed_csv/` while a periodic Tk callback (`_streaming_tick`, every 500 ms) reads the newest processed rows and updates plots. `_bfm_rt_window` builds a rolling window of mag/phase, which `_load_bfm_rt_model`/`_prediction_loop` feed through `rtbfm.window_features()` and a loaded joblib bundle (`BFM_MODEL_PATHS = ['bfm_rt_model_nofoil_xgb.joblib', 'bfm_rt_model_nofoil.joblib']`, first one found wins) to produce a live standing/walking probability. When a labeled session ends, `_snapshot_session_to_bfm_dirs` merges the accumulated `live_bfm_pcap`/`live_bfm_raw_csv` chunks into the same session-named files under `bfm_pcap/`/`bfm_raw_csv/`, and separately writes `bfm_real_imag_csv/`/`bfm_mag_phase_csv/` — so the two GUIs' labeled-data output ends up in the same directories even though `main_gui3.py`'s live path is streaming/incremental rather than collect-then-process.

## `BFMapp.py`

A read-only Streamlit explorer over whatever `main_gui.py`/`main_gui3.py` have written to **`bfm_real_imag_csv/`** (`DATA_DIR` at the top of the file). It reconstructs the complex per-subcarrier matrix from the `SCIDX_<n>_Ratio_Real`/`SCIDX_<n>_Ratio_Imag` columns and renders it — temporal magnitude/phase traces, constellation, PCA(n=1), heatmap, spectrogram/Doppler, motion metrics — with scrub/playback controls. It has no model-loading or capture code of its own; it's purely a visualization layer on the same session CSVs the two GUIs produce, useful for sanity-checking a capture or a labeling session.

## Data directories in active use

- `bfm_pcap/`, `bfm_raw_csv/`, `bfm_processed_csv/` — labeled-collection outputs, one file per `<subject>_<activity>_<description>_<timestamp>` session.
- `bfm_real_imag_csv/`, `bfm_mag_phase_csv/` — the merged per-session CSVs both GUIs write at the end of a collection; `bfm_real_imag_csv/` is also what `BFMapp.py` reads.
- `live_bfm_pcap/`, `live_bfm_raw_csv/`, `live_bfm_processed_csv/` — `main_gui3.py`-only streaming scratch space, drained/merged into the directories above when a session ends.

`./data/` (legacy RSSI-PCAP/CSI-UDP output, only touched by `main_gui.py`'s non-BFM tabs) is not part of the BFM pipeline and can be ignored.
