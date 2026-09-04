"""Stage 1 preprocessing: CSV -> per-session resampled arrays cached in an .npz.

Steps
  1. stream the 1.2 GB CSV in chunks, keeping metadata + 234 magnitude and
     234 phase subcarrier columns as float32
  2. per session: sort by time, drop duplicate timestamps
  3. magnitude -> log (BFM ratio magnitudes are heavy-tailed / multiplicative)
     phase     -> unwrapped across subcarriers, then sin/cos-safe wrapping
  4. resample each session onto a uniform 10 Hz grid (nearest-sample hold with
     a gap mask) so that a "window" has a well defined duration in seconds
"""
import numpy as np
import pandas as pd
import time
import os

DATA_DIR = 'bfm_data.csv'
FIXED_FREQUENCY  = 10.0        # uniform resample rate (median raw rate ~10.25 Hz)
SAVE_DIR = 'sessions_10hz.npz'

t0 = time.time()
head = pd.read_csv(DATA_DIR, nrows=1)
mag_cols = [c for c in head.columns if c.endswith('_Ratio_Mag')]
ph_cols  = [c for c in head.columns if c.endswith('_Ratio_Phase')]
meta_cols = ['session_id', 'environment', 'subject', 'activity', 'timestamp']
scidx = np.array([int(c.split('_')[1]) for c in mag_cols])
order = np.argsort(scidx)
mag_cols = [mag_cols[i] for i in order]
ph_cols  = ['SCIDX_%d_Ratio_Phase' % s for s in scidx[order]]
scidx = scidx[order]

dtypes = {c: np.float32 for c in mag_cols + ph_cols}
parts = []
for ch in pd.read_csv(DATA_DIR, usecols=meta_cols + mag_cols + ph_cols,
                      dtype=dtypes, chunksize=25000):
    parts.append(ch)
    print('  read %d rows (%.0fs)' % (sum(len(p) for p in parts), time.time() - t0), flush=True)
df = pd.concat(parts, ignore_index=True)
del parts
df['timestamp'] = pd.to_datetime(df['timestamp'], format='ISO8601', utc=True)

sessions = {}
# separate dataset into sessions as group
for sid, g in df.groupby('session_id', sort=True):
    # sort the timestamp
    g = g.sort_values('timestamp')
    # change the timestamp to be starting from 0
    # e.g. [0, 0.3, 0.45, 1.2,...]
    t = g['timestamp'].astype('int64').to_numpy() / 1e9
    t -= t[0]
    # remove rows that have duplicated timestamp
    keep = np.concatenate([[True], np.diff(t) > 0])
    g, t = g[keep], t[keep]
    # skip the session if the rows left are less than 50
    if len(t) < 50:
        print('  skip short session', sid, len(t)); continue

    # log-transform the magnitude columns for scaling large values
    # clip with floor value = 1e-6 is to prevent infinite values after log-transformed
    magnitude = np.log(np.clip(g[mag_cols].to_numpy(np.float32), 1e-6, None))
    # Wrap (Relabel) the phase > 2 pi to the range (-pi, pi]
    # by projecting the phase value to a circle and find the angle
    phase = g[ph_cols].to_numpy(np.float32)
    phase = np.angle(np.exp(1j * phase.astype(np.float64))).astype(np.float32)

    # uniform grid, nearest raw sample; mask samples whose nearest neighbour is far
    grid = np.arange(0.0, t[-1] + 1e-9, 1.0 / FIXED_FREQUENCY)
    j = np.searchsorted(t, grid)
    j = np.clip(j, 1, len(t) - 1)

    # grid tick with no real packet nearby (with gap label) will duplicate the mag/phase values of the nearest row
    pick = np.where(np.abs(t[j - 1] - grid) <= np.abs(t[j] - grid), j - 1, j)
    gap = np.abs(t[pick] - grid) > (2.0 / FIXED_FREQUENCY)      # no real packet nearby (gap label)

    # timestamp column is changed to uniform tick with gap label
    sessions[sid] = dict(
        mag=magnitude[pick], phase=phase[pick], gap=gap, t=grid.astype(np.float32),
        activity=g['activity'].iloc[0], environment=g['environment'].iloc[0],
        subject=g['subject'].iloc[0])

print('sessions:', len(sessions), 'gap fraction: %.4f' %
      np.mean([s['gap'].mean() for s in sessions.values()]))

sids = sorted(sessions)
np.savez_compressed(
    SAVE_DIR,
    scidx=scidx,
    sids=np.array(sids),
    activity=np.array([sessions[s]['activity'] for s in sids]),
    environment=np.array([sessions[s]['environment'] for s in sids]),
    subject=np.array([sessions[s]['subject'] for s in sids]),
    lengths=np.array([len(sessions[s]['t']) for s in sids]),
    mag=np.concatenate([sessions[s]['mag'] for s in sids]),
    phase=np.concatenate([sessions[s]['phase'] for s in sids]),
    gap=np.concatenate([sessions[s]['gap'] for s in sids]),
    frequency=np.array([FIXED_FREQUENCY]))
print('wrote %s (%.1f MB) in %.0fs' % (SAVE_DIR, os.path.getsize(SAVE_DIR) / 1e6, time.time() - t0))
