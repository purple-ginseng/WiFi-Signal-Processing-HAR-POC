import pandas as pd
import glob
import plotly.express as px

def plot_3d_subcarriers_plotly(csv_files):
    """
    Reads data from multiple CSV files, and plots the subcarrier data 
    in an interactive 3D graph using Plotly.
    Color represents subcarrier value (heatmap-style), and symbols 
    represent labels.

    Args:
        csv_files (list): A list of paths to the CSV files.
    """

    # Read and concatenate all CSV files into a single DataFrame
    all_data = []
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            all_data.append(df)
        except Exception as e:
            print(f"Error reading {csv_file}: {e}")
            continue

    if not all_data:
        print("No data to plot.")
        return

    combined_df = pd.concat(all_data, ignore_index=True)

    # Prepare data for Plotly: Create one row per subcarrier point
    plot_data = []
    pkt_columns = [f'pkt{i}' for i in range(60)] # Assuming pkt0 to pkt59

    for index, row in combined_df.iterrows():
        label = row['label']
        for i, col_name in enumerate(pkt_columns):
            if col_name in row:
                plot_data.append({
                    'packet_index': index,
                    'subcarrier_index': i,
                    'value': row[col_name],
                    'label': label
                })

    if not plot_data:
        print("No valid 'pkt' data found to plot.")
        return
        
    plotly_df = pd.DataFrame(plot_data)

    # Create the interactive 3D scatter plot
    print("Generating interactive 3D plot...")
    fig = px.scatter_3d(
        plotly_df,
        x='subcarrier_index',
        y='packet_index',
        z='value',
        color='value',  # Color by subcarrier value (heatmap)
        symbol='label', # Use different symbols for each label
        title='Interactive 3D Subcarrier Data (Heatmap Color, Symbol by Label)',
        labels={'value': 'Subcarrier Value', 'packet_index': 'Packet Index', 'subcarrier_index': 'Subcarrier Index'},
        color_continuous_scale=px.colors.sequential.Viridis # You can change the colormap, e.g., 'Plasma', 'Hot'
    )

    # Improve layout and marker size
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=40))
    fig.update_traces(marker=dict(size=2)) # Adjust marker size if needed

    # Show the plot (this will open in your default web browser)
    fig.show()

if __name__ == '__main__':
    # You might need to install plotly first:
    # pip install plotly pandas
    
    # Get all CSV files in the current directory
    csv_files = glob.glob('*.csv')

    if not csv_files:
        print("No CSV files found in the current directory.")
        print("Please ensure your CSV files are in the same folder as this script.")
    else:
        plot_3d_subcarriers_plotly(csv_files)