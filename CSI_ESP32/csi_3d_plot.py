# csi_3d_plot.py
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import glob

# ---------- CORE PLOTTER -------------------------------------------------
def plot_csi_time_surface(
    csv_folder: str = ".",
    max_frames: int | None = None,        # limit packets / rows for speed
    as_lines: bool = True,                # True = line-for-each-frame, False = surface
    opacity: float = 0.65,
    camera_eye: tuple[float, float, float] = (1.6, 1.3, 0.6),
):
    """
    Create an interactive 3-D view of CSI magnitudes vs. time and sub-carrier.

    Expected CSV columns:  timestamp, subcarrier_index, magnitude
    ───────────────────────────────────────────────────────────────────────
    * `csv_folder` : directory that holds one or more *_log.csv files
    * `max_frames` : sample first N unique timestamps (None = all)
    * `as_lines`   : True → 3-D spaghetti plot, False → surface heat-map
    """

    # 1️⃣  Load and concat all CSVs in the folder
    paths = glob.glob(f"./{csv_folder}/*.csv")
    if not paths:
        raise FileNotFoundError(f"No CSVs found in “{csv_folder}”")
    df = pd.concat(map(pd.read_csv, paths), ignore_index=True)

    # 2️⃣  Ensure correct dtypes
    df["timestamp"]        = pd.to_numeric(df["timestamp"], errors="coerce")
    df["subcarrier_index"] = pd.to_numeric(df["subcarrier_index"], errors="coerce").astype(int)
    df["magnitude"]        = pd.to_numeric(df["magnitude"], errors="coerce")

    # 3️⃣  Optionally trim to first N frames to keep plot light-weight
    if max_frames is not None:
        first_ts = sorted(df["timestamp"].unique())[:max_frames]
        df = df[df["timestamp"].isin(first_ts)]

    # 4️⃣  Build the figure
    if as_lines:
        fig = go.Figure()
        palette = px.colors.qualitative.Dark24
        for i, ts in enumerate(sorted(df["timestamp"].unique())):
            sub = df[df["timestamp"] == ts]
            fig.add_trace(
                go.Scatter3d(
                    x=sub["timestamp"],
                    y=sub["subcarrier_index"],
                    z=sub["magnitude"],
                    mode="lines",
                    line=dict(color=palette[i % len(palette)], width=1.2),
                    name=f"{ts:0.2f}s",
                    opacity=opacity,
                    hovertemplate=(
                        "t = %{x:.2f}s<br>"
                        "SC = %{y}<br>"
                        "Mag = %{z:.2f}<extra></extra>"
                    ),
                    showlegend=False,
                )
            )
    else:
        # Pivot to wide matrix (rows = sub-carrier, cols = time)
        surf = (df.pivot_table(index="subcarrier_index",
                               columns="timestamp",
                               values="magnitude",
                               aggfunc="mean")
                  .sort_index())
        fig = go.Figure(
            data=go.Surface(
                x=surf.columns,          # time
                y=surf.index,            # sub-carrier
                z=surf.values,
                colorscale="Viridis",
                showscale=True,
                opacity=opacity,
            )
        )

    # 5️⃣  Layout polish
    fig.update_layout(
        title="CSI magnitude vs. Time & Sub-carrier",
        scene=dict(
            xaxis_title="Time (s)",
            yaxis_title="Sub-carrier",
            zaxis_title="Magnitude",
            camera=dict(eye=dict(x=camera_eye[0],
                                 y=camera_eye[1],
                                 z=camera_eye[2])),
        ),
        margin=dict(l=0, r=0, b=0, t=40),
    )
    fig.show()


# ---------- HANDY MAIN ---------------------------------------------------
if __name__ == "__main__":
    plot_csi_time_surface(
        csv_folder=".",          # path to your log files
        max_frames=200,          # None = all
        as_lines=True,           # spaghetti plot
        opacity=0.6,
        camera_eye=(1.8, 1.2, 0.8),
    )
