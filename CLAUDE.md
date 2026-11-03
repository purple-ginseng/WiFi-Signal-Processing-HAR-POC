# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Device-free human activity recognition (HAR) from IEEE 802.11ac/ax **Compressed Beamforming Feedback Matrix (BFM)** reports captured off an OpenWrt router (`tcpdump` on a monitor-mode `mon0` interface). BFM angles (phi/psi) are decompressed into per-subcarrier complex channel ratios, turned into magnitude/phase features, and classified in real time (standing/walking) with a lightweight gradient-boosted-tree model.

**BFM is the current research focus** — the repo also contains an older RSSI/CSI-based approach (`CSIapp.py`, `CSI_MISO_*`, `CSI_SIMO_*`, `RSSIapp.py`, `archive/`, and legacy tabs inside `main_gui.py` itself) that has been superseded; don't extend or "fix" the RSSI/CSI path unless explicitly asked.

**Active surface**: only `main_gui.py`, `main_gui3.py`, and `BFMapp.py` (plus their shared `bfmtool/` and `rtbfm.py` dependencies) are in current use. Everything else at the repo root — legacy CSI/RSSI scripts, `main_gui2.py`, `main_gui copy.py`, the per-environment CNN notebooks/checkpoints (`best_bfm_model_*.keras`, `train_bfm_*.ipynb`), `pages/`, `compare.py`, `train_final.py`, `prep.py`, `archive/` — is kept only for reference and possible future research directions. Don't build on or "clean up" that material unless the user explicitly points at it.

Full setup/hardware/SOP docs (router prep, OpenWrt package install, troubleshooting table, cold-start SOP) live in `README.md` — read it before doing any live-capture work; it is not duplicated here. Note: the README describes `pages/live_activity_detection.py` as the live-detection entry point, but that page is **not** part of the active surface — `main_gui3.py` is its Tk equivalent and is what's actually maintained.

For a breakdown of `bfmtool/`'s internal pipeline, the `main_gui.py`/`main_gui3.py` implementation internals, `BFMapp.py`, and the data-directory layout, load the `bfm-architecture` skill — that detail is reconstructable from the code and isn't needed on every turn.

## Running things

No build step or linter exists in this repo. There is a pytest suite (see **Tests** below) covering everything that doesn't need the router; live-capture behaviour still has to be verified by running the relevant GUI against the real hardware.

```bash
pip install -r requirements.txt

# Labeled BFM data collection (older Tk GUI; also has legacy, non-BFM tabs)
python main_gui.py

# Streaming BFM collection + real-time standing/walking detection (current Tk GUI)
python main_gui3.py

# Offline BFM CSV explorer / signal-processing stage viewer (Streamlit)
streamlit run BFMapp.py
```

This machine is memory-tight (TensorFlow + a live GUI + SSH capture won't comfortably coexist). Run heavy jobs one at a time and stop background jobs as soon as they're done.

## Tests

`tests/` is a pytest suite for `main_gui.py` and `main_gui3.py` that runs with **no router**: SSH/tcpdump/iperf3 are replaced by fakes, filesystem output goes to a temp cwd, and `messagebox` dialogs are recorded instead of shown. It covers the session-naming helpers, `convert_real_imag_to_mag_phase`, `main_gui._merge_bfm_session_csv`, `main_gui3.LiveDataCollector` (buffer accessors, capture-tail wait, and the processing thread driven on a real pcap in `tests/fixtures/`), the real-time detector (`_bfm_rt_window`, `_prediction_loop`, and the `rtbfm` ↔ deployed-bundle invariant), and the Tk-level setup/close/collect state machines of both GUIs.

**Run the suite after any change to `main_gui.py`, `main_gui3.py`, `rtbfm.py`, or `bfmtool/`, before reporting the change as done.** A failing test is a finding to report, not something to skip.

```bash
# Linux, preferred: headless, no window flashing, ~20 s
xvfb-run -a python -m pytest

# Real display (macOS / no Xvfb); Tk windows are created withdrawn
python -m pytest

# No display at all (or a quick pre-check): skip the Tk tests, ~8 s
python -m pytest -m "not gui"
```

The `gui` tests skip automatically when no display can be opened. The `tshark` tests need Wireshark's `tshark` at the platform path `main_gui3.TSHARK_PATH` and skip otherwise. Adding a test: put shared fakes/fixtures in `tests/conftest.py` (`FakeCollector`, `FakeModel`, `make_ratio_frame`, `make_packet_rows`, `gui_app`/`gui3_app`, `pump_until`); anything that writes to `bfm_*`/`live_bfm_*` dirs must use a fixture that `chdir`s into `tmp_path`. Naming/conversion helpers that exist in both GUI files are parametrized over both modules — keep it that way so the duplicated code can't drift.

## Landmine: the real-time feature/model invariant

`rtbfm.window_features()` (imported, not copied, by `main_gui3.py`) must stay byte-for-byte identical to whatever produced the deployed `bfm_rt_model_nofoil*.joblib` bundle — the bundle pins `fs`/window/hop config but **not** the feature code itself. If you change `rtbfm.py`, the deployed model needs retraining (via `train_final.py`, a reference-only file) or live predictions silently go stale/wrong with no error. `tests/test_main_gui3_rt_detector.py::test_deployed_bundle_matches_rtbfm_features` catches feature-count/name drift, but not a semantic change that keeps the same 43 names — that still needs a retrain.

## GUI disambiguation

When asked to fix "the GUI" without a filename, check which behavior is being described (BFM collection UI vs. live BFM detection vs. the legacy RSSI/CSI tabs) to pick the right file. `main_gui.py` and `main_gui3.py`'s BFM collection code is largely **duplicated, not shared** — a collection/naming fix usually needs to land in both files.

### Coding Instructions
- Keep every long comments to be **within 20 words**
