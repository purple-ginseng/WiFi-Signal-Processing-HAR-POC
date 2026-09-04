"""Shared feature extraction for real-time standing/walking detection from BFM data.

Design constraints (real-time):
  * a window only ever looks backwards -> no future leakage, latency == window width
  * features are cheap: detrend + rFFT over 234 subcarriers, O(T*S*log T)
  * every feature is scale-free where possible (power *fractions*, ratios), because
    the absolute channel-fluctuation level changes a lot between environments
    (foil ~2x nofoil/open) and would otherwise be learnt as an environment cue.
"""
import numpy as np

BANDS = [(0.0, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 3.0), (3.0, 5.0)]


def _agg(feature):
    """Summarise a per-subcarrier statistic across the 234 subcarriers."""
    return [feature.mean(), np.median(feature), feature.std(), np.percentile(feature, 10), np.percentile(feature, 90)]


def window_features(mag, phase, fs):
    """X: (T,S) log-magnitude, P: (T,S) wrapped phase. Returns 1-D float32 vector."""
    timestamp_length = mag.shape[0]
    t = np.linspace(-1, 1, timestamp_length)[:, None]

    # remove the static channel + any linear drift in magnitude
    mag_c = mag - mag.mean(0, keepdims=True)
    mag_c = mag_c - t * (mag_c * t).sum(0, keepdims=True) / (t * t).sum()

    # std(0) instead of std() to tell the function collapse for each column (subcarrier)
    # so that we can get one value for each subcarrier for one window
    mag_std = mag_c.std(0) + 1e-8
    mag_absolute_difference = np.abs(np.diff(mag_c, axis=0)).mean(0)
    mag_roughness = mag_absolute_difference / mag_std

    # find wrapped first differences of phase
    # find differences between phase can caused unwrapped phase again
    # so we need to do phase wrapping again here
    phase_difference = np.angle(np.exp(1j * np.diff(phase.astype(np.float64), axis=0)))
    phase_absolute_difference = np.abs(phase_difference).mean(0)
    phase_std = phase_difference.std(0) + 1e-8

    # 25 Features: 5 x Mag_Std, Mag_Absolute_Difference, Mag_Roughness, Phase_Absolute_Difference, Phase_Roughness
    feats = _agg(mag_std) + _agg(mag_absolute_difference) + _agg(mag_roughness) + _agg(phase_absolute_difference) + _agg(phase_absolute_difference / phase_std)

    # convert from value at each time to content at each frequency, preserving information
    # compute power for each frequency bin (band), use fraction of power as feature
    w = np.hanning(timestamp_length)[:, None] # to smoothly join the repeated 20 samples in rfft
    f1 = np.fft.rfft(mag_c * w, axis=0)
    power = (f1.real ** 2 + f1.imag ** 2)
    f2 = np.fft.rfftfreq(timestamp_length, 1 / fs)
    total_power = power.sum(0) + 1e-12
    frac = np.stack([power[(f2 >= low) & (f2 < high)].sum(0) / total_power for low, high in BANDS])
    for k in range(frac.shape[0]):
        feats += [frac[k].mean(), np.median(frac[k]), frac[k].std()]

    # magnitude correlation within subcarriers
    subcarrier_num = mag_c.shape[1]
    z_score = mag_c / mag_std
    correlation = (z_score.T @ z_score) / timestamp_length
    iu = np.triu_indices(subcarrier_num, 1) # only extract the non-diagonal pairs
    cv = correlation[iu]
    feats += [cv.mean(), np.abs(cv).mean(), cv.std()] # np.abs() to prevent + and - cancellation

    return np.asarray(feats, dtype=np.float32)


FEATURE_NAMES = None  # filled lazily by feature_names()


def feature_names():
    a = lambda p: ['%s_%s' % (p, s) for s in ('mean', 'med', 'std', 'p10', 'p90')]
    # the five per-subcarrier statistics, each summarised 5 ways by _agg
    n = a('sd') + a('absdiff') + a('rough') + a('phdiff') + a('phrough')
    # the band power fractions
    for low, high in BANDS:
        n += ['pfrac%.1f-%.1f_%s' % (low, high, s) for s in ('mean', 'med', 'std')]
    # the cross-subcarrier correlation summary
    n += ['corr_mean', 'corr_absmean', 'corr_std']
    return n


# Load the variable length data saved from prep.py
def load_sessions(path='sessions_10hz.npz', envs=None):
    """envs: iterable of environments to keep. Defaults to nofoil only"""
    # decide which environments to include
    if envs is None:
        envs = ['nofoil']
    data = np.load(path)
    session_lengths = data['lengths']
    # calculate the cumulative sum of elements in lengths array,
    # which eventually equals to the index to separate the sessions
    seps = np.concatenate([[0], np.cumsum(session_lengths)])
    out = []
    for i in range(len(session_lengths)):
        session_length = slice(seps[i], seps[i + 1]) # equivalent to off[i]:off[i+1]
        if envs is not None and str(data['environment'][i]) not in envs:
            continue
        out.append(dict(sid=str(data['sids'][i]), mag=data['mag'][session_length],
                        phase=data['phase'][session_length], gap=data['gap'][session_length],
                        activity=str(data['activity'][i]), environment=str(data['environment'][i]),
                        subject=str(data['subject'][i])))
    # older caches (sessions_10hz.npz as shipped) name the rate 'fs'
    rate = data['frequency'] if 'frequency' in data.files else data['fs']
    return out, float(rate[0])


def build_windows(sessions, fixed_frequency, win_s, hop_s=0.5, max_gap=0.15):
    """Slice every session into windows and create features using the windows"""
    window_size = int(round(win_s * fixed_frequency)) # e.g. 20 samples
    hop_size = max(1, int(round(hop_s * fixed_frequency))) # e.g. 5 samples
    X, y, session, env, subj, end_time = [], [], [], [], [], []
    for s in sessions:
        n = len(s['mag']) # number of samples in a session
        for a in range(0, n - window_size + 1, hop_size):
            b = a + window_size # e.g. (a, b) => (0, 20)
            # if more than 15% of the row have gap label,
            # the window is skipped
            if s['gap'][a:b].mean() > max_gap:
                continue

            # If window is accepted, create features for that window
            X.append(window_features(s['mag'][a:b], s['phase'][a:b], fixed_frequency))
            y.append(1 if s['activity'] == 'walking' else 0)
            session.append(s['sid'])
            env.append(s['environment'])
            subj.append(s['subject'])
            end_time.append(b / fixed_frequency)
    return (np.asarray(X), np.asarray(y), np.asarray(session),
            np.asarray(env), np.asarray(subj), np.asarray(end_time))
