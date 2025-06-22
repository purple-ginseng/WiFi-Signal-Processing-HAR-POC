import glob
import pandas as pd
import numpy as np

import plotly.express as px
import plotly.graph_objects as go

def plot_3d_csi_lines_refined_groups(
    csv_folder="",
    sample_per_group=100,
    line_width=1.0,
    opacity=0.6,
    camera_eye=(1.5, 1.5, 0.6)
):
    # 1) Load + concat
    files = glob.glob(f"{csv_folder}/*.csv")
    if not files:
        raise FileNotFoundError(f"No CSVs found in {csv_folder}")
    df = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)
    
    # 2) Identify groups: only SCIDX_ columns with psi or phi
    psi_cols = [c for c in df.columns if c.startswith("SCIDX_") and 'psi' in c]
    phi_cols = [c for c in df.columns if c.startswith("SCIDX_") and 'phi' in c]
    
    # Safely coerce each group
    if psi_cols:
        df[psi_cols] = df[psi_cols].apply(pd.to_numeric, errors="coerce")
    if phi_cols:
        df[phi_cols] = df[phi_cols].apply(pd.to_numeric, errors="coerce")
    
    # 3) Assign a packet index (0…N-1)
    df = df.reset_index().rename(columns={"index": "packet_index"})
    
    # 4) Sample up to `sample_per_group` packets
    df_samp = df.sample(n=min(len(df), sample_per_group), random_state=1)
    
    # 5) Define groups
    groups = {
        "PSI": psi_cols,
        "PHI": phi_cols
    }
    
    color_map = px.colors.qualitative.Plotly
    
    for group_idx, (group_name, cols) in enumerate(groups.items()):
        if not cols:
            print(f"Skipping group {group_name} (no matching columns)")
            continue
        
        long = df_samp.melt(
            id_vars=["packet_index"],
            value_vars=cols,
            var_name="subcarrier_index",
            value_name="value"
        )
        long = long.sort_values(["packet_index", "subcarrier_index"])
        
        # Create separate figure for each group
        fig = go.Figure()
        for i, pkt_idx in enumerate(long["packet_index"].unique()):
            subgrp = long[long["packet_index"] == pkt_idx]
            fig.add_trace(
                go.Scatter3d(
                    x=subgrp["packet_index"],         
                    y=subgrp["subcarrier_index"],     
                    z=subgrp["value"],                
                    mode="lines",
                    line=dict(
                        width=line_width,
                        color=color_map[i % len(color_map)]
                    ),
                    opacity=opacity,
                    name=f"Packet {pkt_idx}",
                    legendgroup="packets",
                    showlegend=False,
                    hovertemplate=(
                        f"Group: {group_name}<br>"
                        "Packet: %{x}<br>"
                        "Subcarrier: %{y}<br>"
                        "CSI: %{z:.1f}"
                        "<extra></extra>"
                    )
                )
            )
        
        fig.update_layout(
            title=f"3D CSI Lines - {group_name} Group",
            scene=dict(
                xaxis=dict(title="Packet #"),
                yaxis=dict(title="Subcarrier"),
                zaxis=dict(title="Amplitude"),
                camera=dict(eye=dict(x=camera_eye[0], y=camera_eye[1], z=camera_eye[2]))
            ),
            margin=dict(l=0, r=0, b=0, t=40)
        )
        
        fig.show()


if __name__ == "__main__":
    plot_3d_csi_lines_refined_groups(
        csv_folder=".",
        sample_per_group=150,
        line_width=1.2,
        opacity=0.5,
        camera_eye=(1.8, 1.2, 0.8)
    )
