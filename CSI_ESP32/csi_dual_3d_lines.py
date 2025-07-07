import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

def plot_dual_csi_lines(csv_path: str, max_lines: int = 200):
    # Load and sanitize
    df = pd.read_csv(csv_path)
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    df["subcarrier_index"] = pd.to_numeric(df["subcarrier_index"], errors="coerce").astype(int)
    df["magnitude"] = pd.to_numeric(df["magnitude"], errors="coerce")
    df["phase"] = pd.to_numeric(df["phase"], errors="coerce")
    df = df.dropna(subset=["timestamp", "subcarrier_index", "magnitude", "phase"])

    # Limit to first N time slices
    unique_times = sorted(df["timestamp"].unique())[:max_lines]
    df = df[df["timestamp"].isin(unique_times)]

    palette = px.colors.qualitative.Dark24

    # === Magnitude Plot ===
    fig_mag = go.Figure()
    for i, t in enumerate(unique_times):
        subdf = df[df["timestamp"] == t]
        fig_mag.add_trace(go.Scatter3d(
            x=subdf["timestamp"],
            y=subdf["subcarrier_index"],
            z=subdf["magnitude"],
            mode="lines",
            line=dict(color=palette[i % len(palette)], width=1.5),
            name=f"{t:.2f}s",
            showlegend=False,
            hovertemplate="t=%{x:.2f}s<br>SC=%{y}<br>Mag=%{z:.2f}<extra></extra>"
        ))
    fig_mag.update_layout(
        title="CSI Magnitude over Time and Subcarriers",
        scene=dict(
            xaxis_title="Time (s)",
            yaxis_title="Subcarrier",
            zaxis_title="Magnitude"
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )

    # === Phase Plot ===
    fig_phase = go.Figure()
    for i, t in enumerate(unique_times):
        subdf = df[df["timestamp"] == t]
        fig_phase.add_trace(go.Scatter3d(
            x=subdf["timestamp"],
            y=subdf["subcarrier_index"],
            z=subdf["phase"],
            mode="lines",
            line=dict(color=palette[i % len(palette)], width=1.5),
            name=f"{t:.2f}s",
            showlegend=False,
            hovertemplate="t=%{x:.2f}s<br>SC=%{y}<br>Phase=%{z:.1f}°<extra></extra>"
        ))
    fig_phase.update_layout(
        title="CSI Phase over Time and Subcarriers",
        scene=dict(
            xaxis_title="Time (s)",
            yaxis_title="Subcarrier",
            zaxis_title="Phase (°)"
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )

    # === Show Both Plots ===
    fig_mag.show()
    fig_phase.show()

if __name__ == "__main__":
    plot_dual_csi_lines("CSI_ESP32/csi_log.csv", max_lines=10000)
