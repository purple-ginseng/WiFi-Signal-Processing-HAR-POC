import glob
import pandas as pd
import numpy as np

import plotly.express as px         # ← add this
import plotly.graph_objects as go

def plot_3d_csi_lines_refined(
    csv_folder="./data",
    sample_per_label=100,
    line_width=1.0,
    opacity=0.6,
    camera_eye=(1.5, 1.5, 0.6)
):
    # 1) Load + concat
    files = glob.glob(f"{csv_folder}/*.csv")
    if not files:
        raise FileNotFoundError(f"No CSVs found in {csv_folder}")
    df = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)
    
    # 2) Identify & coerce pkt columns
    pkt_cols = [c for c in df.columns if c.startswith("pkt")]
    df[pkt_cols] = df[pkt_cols].apply(pd.to_numeric, errors="coerce")
    
    # 3) Assign a packet index (0…N-1)
    df = df.reset_index().rename(columns={"index": "packet_index"})
    
    # 4) Sample up to `sample_per_label` packets per label
    samples = []
    for lbl, grp in df.groupby("label"):
        samples.append(grp.sample(n=min(len(grp), sample_per_label), random_state=1))
    df_samp = pd.concat(samples, ignore_index=True)
    
    # 5) Melt wide→long
    long = df_samp.melt(
        id_vars=["packet_index", "label"],
        value_vars=pkt_cols,
        var_name="subcarrier_index",
        value_name="value"
    )
    long["subcarrier_index"] = (
        long["subcarrier_index"]
            .str.replace("pkt", "", regex=False)
            .astype(int)
    )
    long = long.sort_values(["label", "packet_index", "subcarrier_index"])
    
    # 6) Build the figure
    fig = go.Figure()
    color_map = px.colors.qualitative.Plotly   # now px is defined
    for i, lbl in enumerate(long["label"].unique()):
        grp = long[long["label"] == lbl]
        
        # For each sampled packet, add one Scatter3d trace
        for pkt_idx, subgrp in grp.groupby("packet_index"):
            fig.add_trace(
                go.Scatter3d(
                    x=subgrp["packet_index"],         # Packet → X
                    y=subgrp["subcarrier_index"],     # Subcarrier → Y
                    z=subgrp["value"],                # CSI → Z
                    mode="lines",
                    line=dict(
                        width=line_width,
                        color=color_map[i % len(color_map)]
                    ),
                    opacity=opacity,
                    name=str(lbl),
                    legendgroup=str(lbl),
                    showlegend=True,
                    hovertemplate=(
                        "Packet: %{x}<br>"
                        "Subcarrier: %{y}<br>"
                        "CSI: %{z:.1f}"
                        "<extra></extra>"
                    )
                )
            )
    
    # 7) Tidy up layout
    fig.update_layout(
        title="3D Lines per Packet",
        scene=dict(
            xaxis=dict(title="Packet #"),
            yaxis=dict(title="Subcarrier #"),
            zaxis=dict(title="Amplitude"),
            camera=dict(eye=dict(x=camera_eye[0], y=camera_eye[1], z=camera_eye[2]))
        ),
        legend=dict(title="Activity Label", yanchor="top", y=0.95, xanchor="left", x=0.02),
        margin=dict(l=0, r=0, b=0, t=40)
    )
    
    fig.show()


if __name__ == "__main__":
    # pip install pandas plotly
    plot_3d_csi_lines_refined(
        csv_folder="./data",
        sample_per_label=150,
        line_width=1.2,
        opacity=0.5,
        camera_eye=(1.8, 1.2, 0.8)
    )
