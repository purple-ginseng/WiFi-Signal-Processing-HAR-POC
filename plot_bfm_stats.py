import glob
import pandas as pd
import numpy as np

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


@st.cache_data(show_spinner=False)
def load_csv_folder(csv_folder):
    files = glob.glob(f"{csv_folder}/*.csv")
    if not files:
        raise FileNotFoundError(f"No CSVs found in {csv_folder}")
    frames = [pd.read_csv(f) for f in files]
    # Drop empty / all-NA frames so concat dtype inference stays stable
    frames = [f for f in frames if not f.empty and not f.isna().all().all()]
    if not frames:
        raise ValueError(f"All CSVs in {csv_folder} were empty or all-NA")
    return pd.concat(frames, ignore_index=True)


def build_group_figures(
    csv_folder="",
    sample_per_group=100,
    line_width=1.0,
    opacity=0.6,
    camera_eye=(1.5, 1.5, 0.6),
):
    # 1) Load + concat
    df = load_csv_folder(csv_folder)

    # 2) Identify groups: only SCIDX_ columns with psi or phi
    psi_cols = [c for c in df.columns if c.startswith("SCIDX_") and "psi" in c]
    phi_cols = [c for c in df.columns if c.startswith("SCIDX_") and "phi" in c]

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
        "PHI": phi_cols,
    }

    color_map = px.colors.qualitative.Plotly
    figures = {}

    for group_name, cols in groups.items():
        if not cols:
            figures[group_name] = None
            continue

        long = df_samp.melt(
            id_vars=["packet_index"],
            value_vars=cols,
            var_name="subcarrier_index",
            value_name="value",
        )
        long = long.sort_values(["packet_index", "subcarrier_index"])

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
                        color=color_map[i % len(color_map)],
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
                    ),
                )
            )

        fig.update_layout(
            title=f"3D CSI Lines - {group_name} Group",
            scene=dict(
                xaxis=dict(title="Packet #"),
                yaxis=dict(title="Subcarrier"),
                zaxis=dict(title="Amplitude"),
                camera=dict(
                    eye=dict(x=camera_eye[0], y=camera_eye[1], z=camera_eye[2])
                ),
            ),
            margin=dict(l=0, r=0, b=0, t=40),
            height=700,
        )
        figures[group_name] = fig

    return figures


def main():
    st.set_page_config(page_title="BFM CSI 3D Lines", layout="wide")
    st.title("3D CSI Lines — BFM Stats")

    with st.sidebar:
        st.header("Settings")
        csv_folder = st.text_input("CSV folder", value=".")
        sample_per_group = st.slider("Samples per group", 10, 1000, 150, step=10)
        line_width = st.slider("Line width", 0.5, 5.0, 1.2, step=0.1)
        opacity = st.slider("Opacity", 0.1, 1.0, 0.5, step=0.05)
        eye_x = st.slider("Camera eye X", 0.5, 3.0, 1.8, step=0.1)
        eye_y = st.slider("Camera eye Y", 0.5, 3.0, 1.2, step=0.1)
        eye_z = st.slider("Camera eye Z", 0.1, 3.0, 0.8, step=0.1)

    try:
        figures = build_group_figures(
            csv_folder=csv_folder,
            sample_per_group=sample_per_group,
            line_width=line_width,
            opacity=opacity,
            camera_eye=(eye_x, eye_y, eye_z),
        )
    except (FileNotFoundError, ValueError) as e:
        st.error(str(e))
        return

    for group_name, fig in figures.items():
        if fig is None:
            st.info(f"Skipping group {group_name} (no matching columns)")
            continue
        st.subheader(f"{group_name} Group")
        st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
