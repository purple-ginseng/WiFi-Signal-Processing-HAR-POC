"""
Real-time Activity Recognition Testing with Matthew Dataset

This page simulates real-time activity detection by processing WiFi BFM data
in sliding windows of 50 packets, displaying predictions for:
- Walking
- Standing
- Detecting (initial state)
"""

import time
from pathlib import Path

import joblib

# Configure matplotlib BEFORE importing pyplot for thread safety
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for thread safety
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from sklearn.decomposition import PCA
from scipy.ndimage import median_filter

# Disable matplotlib warnings
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')

# TensorFlow on macOS (Metal) can be unstable for light inference; force CPU for reliability.
try:
    tf.config.experimental.set_visible_devices([], "GPU")
except Exception:
    pass

# Configuration
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "bfm_processed_csv"
MODEL_PATH = BASE_DIR / "best_bfm_model_open.keras"  # Using nofoil model since Matthew data is alt_nofoil
PCA_ARTIFACT_PATH = BASE_DIR / "bfm_open_pca.pkl"
WINDOW_SIZE = 50  # Match model training window
UPDATE_INTERVAL = 1.0  # 1 second between updates
PCA_COMPONENTS = 5  # Match training configuration
ENVIRONMENT_FILTER = "open"
TRAIN_SUBJECTS = ['abel', 'collin', 'ivan', 'kenny']
MEDIAN_FILTER_SIZE = 9
ROLLING_MEAN_WINDOW = 20
ROLLING_STD_WINDOW = 101

# Activity labels
ACTIVITY_LABELS = {
    0: 'Standing',
    1: 'Walking'
}


@st.cache_resource
def load_model():
    """Load the trained BFM model"""
    if not MODEL_PATH.exists():
        st.error(f"Model file not found: {MODEL_PATH}")
        return None

    try:
        model = tf.keras.models.load_model(str(MODEL_PATH))
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


def apply_median_filter_sessions(df, feature_cols, session_col='session_id', window_size=MEDIAN_FILTER_SIZE):
    """Median-filter features per session to remove outliers."""
    if df.empty:
        return df

    df_filtered = df.copy()
    for session_id, idx in df.groupby(session_col).groups.items():
        session_data = df.loc[idx, feature_cols].to_numpy(dtype=np.float32, copy=True)
        filtered = median_filter(session_data, size=(window_size, 1), mode='nearest')
        df_filtered.loc[idx, feature_cols] = filtered
    return df_filtered


def apply_rolling_z_score(df, feature_cols, session_col='session_id',
                          mean_window=ROLLING_MEAN_WINDOW, std_window=ROLLING_STD_WINDOW):
    """Apply rolling z-score normalization per session on PCA features."""
    if df.empty:
        return df

    normalized_parts = []
    for session_id, session_df in df.groupby(session_col):
        features = session_df[feature_cols]
        rolling_mean = features.rolling(window=mean_window, min_periods=1).mean()
        centered = features - rolling_mean
        rolling_std = centered.rolling(window=std_window, min_periods=std_window).std()
        normalized = centered / (rolling_std + 1e-8)
        normalized = normalized.dropna()

        if normalized.empty:
            fallback = centered.fillna(0.0)
            trimmed = session_df.copy()
            trimmed[feature_cols] = fallback
            normalized_parts.append(trimmed)
        else:
            trimmed = session_df.loc[normalized.index].copy()
            trimmed[feature_cols] = normalized
            normalized_parts.append(trimmed)

    if not normalized_parts:
        return pd.DataFrame(columns=df.columns)

    return pd.concat(normalized_parts, ignore_index=True)


def _compute_feature_columns(data_file):
    """Determine magnitude/phase feature column order from combined dataset."""
    sample = pd.read_csv(data_file, nrows=1)
    mag_cols = sorted([c for c in sample.columns if c.endswith("Ratio_Mag")])
    phase_cols = sorted([c for c in sample.columns if c.endswith("Ratio_Phase")])
    feature_cols = mag_cols + phase_cols
    return feature_cols, mag_cols, phase_cols


def _fit_pca_artifacts():
    """Fit PCA on training subjects for the target environment and cache artifacts."""
    data_file = BASE_DIR / "Cleaned and combined" / "bfm_data.csv"
    if not data_file.exists():
        st.error(f"Training data file not found: {data_file}")
        return None

    feature_cols, mag_cols, phase_cols = _compute_feature_columns(data_file)
    usecols = ['session_id', 'environment', 'subject'] + feature_cols

    df = pd.read_csv(data_file, usecols=usecols)
    df['environment'] = df['environment'].str.lower()
    df = df[df['environment'] == ENVIRONMENT_FILTER]

    if df.empty:
        st.error(f"No data found for environment '{ENVIRONMENT_FILTER}' in {data_file}")
        return None

    df = apply_median_filter_sessions(df, feature_cols, window_size=MEDIAN_FILTER_SIZE)

    train_mask = df['subject'].str.lower().isin(TRAIN_SUBJECTS)
    train_df = df.loc[train_mask, feature_cols].astype(np.float32, copy=False)
    train_df = train_df.replace([np.inf, -np.inf], np.nan).dropna()

    if train_df.empty:
        st.error("Training subset for PCA is empty after preprocessing.")
        return None

    pca = PCA(n_components=PCA_COMPONENTS)
    pca.fit(train_df.to_numpy())

    artifacts = {
        "pca": pca,
        "feature_cols": feature_cols,
        "mag_cols": mag_cols,
        "phase_cols": phase_cols,
    }

    try:
        joblib.dump(artifacts, PCA_ARTIFACT_PATH)
    except Exception as exc:
        st.warning(f"Could not cache PCA artifacts to {PCA_ARTIFACT_PATH}: {exc}")

    return artifacts


@st.cache_resource
def load_preprocessing_artifacts():
    """Load cached PCA and feature column metadata; train if missing."""
    if PCA_ARTIFACT_PATH.exists():
        try:
            artifacts = joblib.load(PCA_ARTIFACT_PATH)
            if isinstance(artifacts, dict) and {"pca", "feature_cols", "mag_cols", "phase_cols"} <= artifacts.keys():
                return artifacts
        except Exception as exc:
            st.warning(f"Failed to load cached PCA artifacts: {exc}")

    return _fit_pca_artifacts()


def convert_real_imag_to_mag_phase(df, mag_cols, phase_cols):
    """Convert Ratio Real/Imag columns to magnitude and phase features."""
    feature_data = {}

    for mag_col in mag_cols:
        base = mag_col.replace("_Ratio_Mag", "")
        real_col = f"{base}_Ratio_Real"
        imag_col = f"{base}_Ratio_Imag"

        if real_col not in df.columns or imag_col not in df.columns:
            feature_data[mag_col] = np.zeros(len(df), dtype=np.float32)
            continue

        real_vals = pd.to_numeric(df[real_col], errors='coerce').astype(np.float32)
        imag_vals = pd.to_numeric(df[imag_col], errors='coerce').astype(np.float32)
        feature_data[mag_col] = np.sqrt(real_vals**2 + imag_vals**2)

    for phase_col in phase_cols:
        base = phase_col.replace("_Ratio_Phase", "")
        real_col = f"{base}_Ratio_Real"
        imag_col = f"{base}_Ratio_Imag"

        if real_col not in df.columns or imag_col not in df.columns:
            feature_data[phase_col] = np.zeros(len(df), dtype=np.float32)
            continue

        real_vals = pd.to_numeric(df[real_col], errors='coerce').astype(np.float32)
        imag_vals = pd.to_numeric(df[imag_col], errors='coerce').astype(np.float32)
        feature_data[phase_col] = np.arctan2(imag_vals, real_vals)

    return pd.DataFrame(feature_data, index=df.index)


@st.cache_data
def load_matthew_data():
    """Load PG standing dataset file"""
    if not DATA_DIR.exists():
        st.error(f"Data directory not found: {DATA_DIR}")
        return None, None

    # Use the specific PG standing dataset
    pg_file = DATA_DIR / "bfm_data_blindtest_pg_alt_open_20251028_185759.csv"

    if not pg_file.exists():
        # Fallback to any pg files if specific file not found
        matthew_files = sorted(DATA_DIR.glob("*pg*.csv"))
        if not matthew_files:
            st.error(f"No PG dataset files found in {DATA_DIR}")
            return None, None
    else:
        matthew_files = [pg_file]
    artifacts = load_preprocessing_artifacts()

    if not artifacts or artifacts.get("pca") is None:
        return None, None

    pca = artifacts["pca"]
    feature_cols = artifacts["feature_cols"]
    mag_cols = artifacts["mag_cols"]
    phase_cols = artifacts["phase_cols"]
    pca_cols = [f"pca{i}" for i in range(PCA_COMPONENTS)]

    sessions = []

    for file_path in matthew_files:
        filename = file_path.name

        # For PG dataset, treat as blind test (no ground truth labels)
        if 'pg' in filename.lower():
            label = 'Unknown'  # Blind test - no ground truth
        elif 'walking' in filename.lower():
            label = 'Walking'
        elif 'standing' in filename.lower():
            label = 'Standing'
        else:
            label = 'Unknown'

        df_raw = pd.read_csv(file_path)

        mag_phase_df = convert_real_imag_to_mag_phase(df_raw, mag_cols, phase_cols)
        mag_phase_df = mag_phase_df.reindex(columns=feature_cols, fill_value=0.0)

        session_id = file_path.stem
        timestamp_series = pd.to_numeric(
            df_raw.get("timestamp", pd.Series(np.arange(len(df_raw), dtype=np.float32))),
            errors='coerce'
        )

        meta_df = pd.DataFrame({
            "session_id": session_id,
            "activity": label,
            "timestamp": timestamp_series
        })

        session_df = pd.concat(
            [meta_df.reset_index(drop=True), mag_phase_df.astype(np.float32).reset_index(drop=True)],
            axis=1
        )
        sessions.append(session_df)

    if not sessions:
        return None, None

    combined = pd.concat(sessions, ignore_index=True)
    combined = apply_median_filter_sessions(combined, feature_cols, window_size=MEDIAN_FILTER_SIZE)

    feature_matrix = combined[feature_cols].to_numpy(dtype=np.float32, copy=True)
    feature_matrix = np.nan_to_num(feature_matrix, nan=0.0, posinf=0.0, neginf=0.0)
    pca_features = pca.transform(feature_matrix)

    pca_df = combined[['session_id', 'activity']].copy()
    for idx, col in enumerate(pca_cols):
        pca_df[col] = pca_features[:, idx]

    normalized = apply_rolling_z_score(pca_df, pca_cols)

    if normalized.empty:
        st.error("Normalization removed all samples; check preprocessing parameters.")
        return None, None

    data_array = normalized[pca_cols].to_numpy(dtype=np.float32, copy=False)
    labels = normalized['activity'].tolist()

    return data_array, labels


def preprocess_window(window_data):
    """Reshape a preprocessed window for model prediction."""
    window_array = np.asarray(window_data, dtype=np.float32)
    return window_array.reshape(1, window_array.shape[0], window_array.shape[1])


def get_activity_color(activity):
    """Return color for activity label"""
    colors = {
        'Standing': '#3498db',  # Blue
        'Walking': '#e74c3c',   # Red
        'Detecting': '#95a5a6'  # Gray
    }
    return colors.get(activity, '#95a5a6')


def plot_prediction_history(history, max_length=20):
    """Plot the recent prediction history"""
    if len(history) == 0:
        return None

    fig, ax = plt.subplots(figsize=(10, 3))

    # Extract data
    timestamps = list(range(len(history)))
    predictions = [h['prediction'] for h in history]
    confidences = [h['confidence'] for h in history]
    true_labels = [h['true_label'] for h in history]

    # Create color mapping
    colors = [get_activity_color(pred) for pred in predictions]

    # Plot bars
    bars = ax.bar(timestamps, confidences, color=colors, alpha=0.7, edgecolor='black')

    # Add true label markers (only if ground truth is available)
    has_ground_truth = any(label != 'Unknown' for label in true_labels)
    if has_ground_truth:
        for ts, true_label in zip(timestamps, true_labels):
            if true_label != 'Unknown':
                marker_color = get_activity_color(true_label)
                ax.scatter(ts, 1.05, marker='v', s=100, color=marker_color,
                          edgecolors='black', linewidths=1.5, zorder=5)

    ax.set_xlabel('Time Step', fontsize=10)
    ax.set_ylabel('Confidence', fontsize=10)

    # Update title based on whether ground truth is available
    if has_ground_truth:
        ax.set_title('Prediction History (▼ = True Label)', fontsize=12, fontweight='bold')
        ax.set_ylim(0, 1.1)
    else:
        ax.set_title('Prediction History (Blind Test)', fontsize=12, fontweight='bold')
        ax.set_ylim(0, 1.05)

    ax.grid(True, alpha=0.3, axis='y')

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=get_activity_color('Standing'), label='Standing'),
        Patch(facecolor=get_activity_color('Walking'), label='Walking'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9)

    plt.tight_layout()
    return fig


def plot_realtime_signal(window_data):
    """Plot the PCA component trajectories within the current window."""
    fig, ax = plt.subplots(figsize=(10, 3))

    for idx in range(window_data.shape[1]):
        ax.plot(window_data[:, idx], label=f'PC {idx + 1}', linewidth=1.5, alpha=0.7)

    ax.set_xlabel('Packet', fontsize=10)
    ax.set_ylabel('Normalized Value', fontsize=10)
    ax.set_title('Current Window (PCA Space)', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def main():
    st.set_page_config("Real-time Activity Test", layout="wide", page_icon="🎯")
    st.title("🎯 Real-time Activity Recognition Test")
    st.caption("Simulating real-time activity detection with PG Standing dataset (bfm_data_pg_standing_home_20251027_211523)")

    # Load model
    with st.spinner("Loading BFM model..."):
        model = load_model()

    if model is None:
        st.error(f"Failed to load model from {MODEL_PATH}")
        st.info("Available model files:")
        model_files = sorted(MODEL_PATH.parent.glob("*.keras"))
        for mf in model_files:
            st.write(f"- {mf}")
        st.stop()

    st.success(f"✅ Model loaded: {MODEL_PATH}")

    # Load data
    with st.spinner("Loading PG dataset..."):
        data, labels = load_matthew_data()

    if data is None:
        st.error("Failed to load PG dataset")
        st.stop()

    st.success(f"✅ Loaded {len(data)} samples from PG dataset")

    # Initialize session state
    if 'rt_index' not in st.session_state:
        st.session_state.rt_index = 0
    if 'rt_playing' not in st.session_state:
        st.session_state.rt_playing = False
    if 'rt_history' not in st.session_state:
        st.session_state.rt_history = []
    if 'rt_speed' not in st.session_state:
        st.session_state.rt_speed = 1.0

    # Sidebar controls
    st.sidebar.markdown("### 🎮 Playback Controls")

    col1, col2, col3 = st.sidebar.columns(3)
    if col1.button("⏮️ Reset", use_container_width=True):
        st.session_state.rt_index = 0
        st.session_state.rt_history = []
        st.session_state.rt_playing = False
        st.rerun()

    play_label = "⏸️ Pause" if st.session_state.rt_playing else "▶️ Play"
    if col2.button(play_label, use_container_width=True):
        st.session_state.rt_playing = not st.session_state.rt_playing
        st.rerun()

    if col3.button("⏭️ Step", use_container_width=True):
        st.session_state.rt_playing = False
        if st.session_state.rt_index + WINDOW_SIZE < len(data):
            st.session_state.rt_index += 1  # Advance by 1 packet (sliding window)
        st.rerun()

    # Speed control
    st.sidebar.markdown("### ⚡ Speed Control")
    speed_options = {
        "0.5x (Slow)": 2.0,
        "1x (Normal)": 1.0,
        "2x (Fast)": 0.5,
        "5x (Very Fast)": 0.2
    }
    speed_label = st.sidebar.select_slider(
        "Playback speed",
        options=list(speed_options.keys()),
        value="1x (Normal)"
    )
    st.session_state.rt_speed = speed_options[speed_label]

    # Position slider
    max_index = max(0, len(data) - WINDOW_SIZE)
    current_index = st.sidebar.slider(
        "Position",
        min_value=0,
        max_value=max_index,
        value=min(st.session_state.rt_index, max_index),
        step=1,  # Step by 1 packet for sliding window
        key="position_slider"
    )

    if current_index != st.session_state.rt_index:
        st.session_state.rt_index = current_index
        st.session_state.rt_playing = False

    # Statistics
    st.sidebar.markdown("### 📊 Statistics")
    total_windows = max_index  # Total possible sliding windows
    current_window = st.session_state.rt_index + 1
    progress = (st.session_state.rt_index / max(1, max_index)) * 100

    st.sidebar.metric("Current Window", f"{current_window} / {total_windows}")
    st.sidebar.metric("Progress", f"{progress:.1f}%")
    st.sidebar.metric("Samples Processed", f"{st.session_state.rt_index + WINDOW_SIZE} / {len(data)}")

    # Main display
    if st.session_state.rt_index + WINDOW_SIZE > len(data):
        st.warning("⚠️ Reached end of dataset")
        st.session_state.rt_playing = False
        st.stop()

    # Extract current window
    window_start = st.session_state.rt_index
    window_end = window_start + WINDOW_SIZE
    current_window_data = data[window_start:window_end]
    current_true_label = labels[window_start] if window_start < len(labels) else 'Unknown'

    # Preprocess and predict
    try:
        window_processed = preprocess_window(current_window_data)

        # Make prediction
        prediction_probs = model.predict(window_processed, verbose=0)[0]
        predicted_class = np.argmax(prediction_probs)
        predicted_activity = ACTIVITY_LABELS.get(predicted_class, 'Unknown')
        confidence = float(prediction_probs[predicted_class])

        # Store in history
        st.session_state.rt_history.append({
            'prediction': predicted_activity,
            'confidence': confidence,
            'true_label': current_true_label,
            'probs': prediction_probs.tolist()
        })

        # Keep only last 20 predictions
        if len(st.session_state.rt_history) > 20:
            st.session_state.rt_history.pop(0)

    except Exception as e:
        st.error(f"Prediction error: {e}")
        st.stop()

    # Display current prediction
    st.markdown("---")
    st.markdown("## 🎯 Current Prediction")

    col1, col2, col3, col4 = st.columns(4)

    # Predicted activity
    activity_color = get_activity_color(predicted_activity)
    col1.markdown(f"### Predicted")
    col1.markdown(f"<h1 style='color: {activity_color};'>{predicted_activity}</h1>",
                  unsafe_allow_html=True)

    # Confidence
    col2.markdown(f"### Confidence")
    col2.markdown(f"<h1 style='color: {'#27ae60' if confidence > 0.7 else '#f39c12'};'>{confidence:.1%}</h1>",
                  unsafe_allow_html=True)

    # True label (only show if known)
    if current_true_label != 'Unknown':
        true_color = get_activity_color(current_true_label)
        col3.markdown(f"### True Label")
        col3.markdown(f"<h1 style='color: {true_color};'>{current_true_label}</h1>",
                      unsafe_allow_html=True)

        # Match indicator
        is_match = predicted_activity == current_true_label
        col4.markdown(f"### Match")
        match_emoji = "✅" if is_match else "❌"
        match_text = "Correct" if is_match else "Wrong"
        col4.markdown(f"<h1>{match_emoji}</h1>", unsafe_allow_html=True)
        col4.markdown(f"<p style='font-size: 20px;'>{match_text}</p>", unsafe_allow_html=True)
    else:
        # Blind test mode
        col3.markdown(f"### Mode")
        col3.markdown(f"<h1 style='color: #95a5a6;'>🔒 Blind Test</h1>",
                      unsafe_allow_html=True)
        col4.markdown(f"### Note")
        col4.markdown(f"<p style='font-size: 16px;'>Ground truth not available</p>", unsafe_allow_html=True)

    # Probability breakdown
    st.markdown("### 📊 Class Probabilities")
    prob_cols = st.columns(len(ACTIVITY_LABELS))
    for idx, (class_id, activity_name) in enumerate(ACTIVITY_LABELS.items()):
        prob = prediction_probs[class_id]
        prob_cols[idx].metric(
            activity_name,
            f"{prob:.1%}",
            delta=None
        )

    # Visualizations
    st.markdown("---")

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("### 📈 Prediction History")
        if st.session_state.rt_history:
            fig_history = plot_prediction_history(st.session_state.rt_history)
            if fig_history:
                st.pyplot(fig_history)
                plt.close(fig_history)  # Prevent memory leak
        else:
            st.info("No history yet. Press Play to start.")

    with col_right:
        st.markdown("### 🌊 Current Signal Window")
        fig_signal = plot_realtime_signal(current_window_data)
        st.pyplot(fig_signal)
        plt.close(fig_signal)  # Prevent memory leak

    # Accuracy statistics
    if len(st.session_state.rt_history) > 0:
        st.markdown("---")
        st.markdown("### 📈 Session Statistics")

        # Check if we have ground truth labels
        has_ground_truth = any(h['true_label'] != 'Unknown' for h in st.session_state.rt_history)

        if has_ground_truth:
            correct = sum(1 for h in st.session_state.rt_history
                         if h['prediction'] == h['true_label'])
            total = len(st.session_state.rt_history)
            accuracy = (correct / total) * 100 if total > 0 else 0
        else:
            # Blind test mode - no accuracy calculation
            correct = 0
            total = len(st.session_state.rt_history)
            accuracy = 0

        if has_ground_truth:
            stat_cols = st.columns(3)
            stat_cols[0].metric("Total Predictions", total)
            stat_cols[1].metric("Correct Predictions", correct)
            stat_cols[2].metric("Accuracy", f"{accuracy:.1f}%")
        else:
            # Blind test mode - show prediction distribution instead
            st.info("🔒 **Blind Test Mode**: Ground truth labels not available. Showing prediction distribution.")

            # Count predictions
            from collections import Counter
            prediction_counts = Counter(h['prediction'] for h in st.session_state.rt_history)

            stat_cols = st.columns(len(ACTIVITY_LABELS) + 1)
            stat_cols[0].metric("Total Predictions", total)

            for idx, (class_id, activity_name) in enumerate(ACTIVITY_LABELS.items()):
                count = prediction_counts.get(activity_name, 0)
                percentage = (count / total * 100) if total > 0 else 0
                stat_cols[idx + 1].metric(
                    activity_name,
                    f"{count}",
                    delta=f"{percentage:.1f}%"
                )

    # Auto-advance if playing
    if st.session_state.rt_playing:
        if st.session_state.rt_index + WINDOW_SIZE < len(data):
            time.sleep(UPDATE_INTERVAL * st.session_state.rt_speed)
            st.session_state.rt_index += 1  # Advance by 1 packet (sliding window)
            st.rerun()
        else:
            st.session_state.rt_playing = False
            st.success("✅ Reached end of dataset!")
            st.balloons()


if __name__ == "__main__":
    main()
