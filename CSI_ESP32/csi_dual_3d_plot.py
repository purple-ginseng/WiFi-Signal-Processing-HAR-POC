import pandas as pd
import plotly.graph_objects as go

CSV_FILE = "./csi_log.csv"
MAX_PACKETS = 150
LINE_WIDTH = 2

# === Load & Clean CSV ===
df = pd.read_csv(CSV_FILE)
df = df.dropna(subset=["timestamp", "subcarrier_index", "magnitude", "phase"])
df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
df["subcarrier_index"] = pd.to_numeric(df["subcarrier_index"], errors="coerce")
df["magnitude"] = pd.to_numeric(df["magnitude"], errors="coerce")
df["phase"] = pd.to_numeric(df["phase"], errors="coerce")

packet_times = sorted(df["timestamp"].unique())[:MAX_PACKETS]
df = df[df["timestamp"].isin(packet_times)]

# === Plot Setup ===
fig = go.Figure()

for i, t in enumerate(packet_times):
    group = df[df["timestamp"] == t]
    color = f"hsl({(i * 360 / MAX_PACKETS) % 360},80%,50%)"

    # Magnitude Line
    fig.add_trace(go.Scatter3d(
        x=[t]*len(group),
        y=group["subcarrier_index"],
        z=group["magnitude"],
        mode="lines",
        line=dict(color=color, width=LINE_WIDTH),
        name=f"Mag t={round(t, 2)}",
        legendgroup="Magnitude",
        showlegend=(i == 0),
        hovertemplate="Time: %{x:.2f}s<br>SC: %{y}<br>Mag: %{z:.2f}<extra></extra>"
    ))

    # Phase Line
    fig.add_trace(go.Scatter3d(
        x=[t]*len(group),
        y=group["subcarrier_index"],
        z=group["phase"],
        mode="lines",
        line=dict(color=color, width=LINE_WIDTH, dash="solid"),
        name=f"Phase t={round(t, 2)}",
        legendgroup="Phase",
        showlegend=(i == 0),
        hovertemplate="Time: %{x:.2f}s<br>SC: %{y}<br>Phase: %{z:.2f}°<extra></extra>"
    ))

# === Layout with Toggles ===
fig.update_layout(
    title="CSI 3D Plot: Magnitude & Phase (Toggle Enabled)",
    scene=dict(
        xaxis_title="Time (s)",
        yaxis_title="Subcarrier Index",
        zaxis_title="Signal Value",
        camera=dict(eye=dict(x=2, y=1.3, z=1.2))
    ),
    legend=dict(title="Signal Type"),
    updatemenus=[dict(
        type="buttons",
        showactive=True,
        buttons=[
            dict(label="Show Both", method="update",
                 args=[{"visible": [True]*2*len(packet_times)},
                       {"title": "CSI 3D Plot: Magnitude & Phase"}]),
            dict(label="Only Magnitude", method="update",
                 args=[{"visible": [True, False]*len(packet_times)},
                       {"title": "CSI 3D Plot: Magnitude Only"}]),
            dict(label="Only Phase", method="update",
                 args=[{"visible": [False, True]*len(packet_times)},
                       {"title": "CSI 3D Plot: Phase Only"}]),
        ],
        direction="right",
        x=0.5,
        xanchor="center",
        y=1.1,
        yanchor="top"
    )],
    margin=dict(l=0, r=0, b=0, t=50)
)

fig.show()
