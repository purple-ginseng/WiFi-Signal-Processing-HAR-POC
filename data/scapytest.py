import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import glob

def plot_3d_subcarriers(csv_files):
    """
    Reads data from multiple CSV files, and plots the subcarrier data in a 3D graph,
    grouping by the 'label' column.

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

    # Get the list of labels
    labels = combined_df['label'].unique()

    # Create a 3D plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot data for each label
    for label in labels:
        label_df = combined_df[combined_df['label'] == label]

        # Get subcarrier data (pkt0 to pkt59)
        subcarrier_data = label_df.loc[:, 'pkt0':'pkt59'].values

        # Create x, y, and z coordinates
        x, y = [], []
        for i in range(subcarrier_data.shape[0]):
            x.extend(range(60))
            y.extend([i] * 60)
        z = subcarrier_data.flatten()

        # Plot the data
        ax.scatter(x, y, z, label=label, s=1)  # s=1 for smaller points

    # Set labels and title
    ax.set_xlabel('Subcarrier Index')
    ax.set_ylabel('Packet Index')
    ax.set_zlabel('Subcarrier Value')
    ax.set_title('3D Subcarrier Data by Activity Label')
    ax.legend()

    # Show the plot
    plt.show()

if __name__ == '__main__':
    # Get all CSV files in the current directory
    csv_files = glob.glob('*.csv')

    if not csv_files:
        print("No CSV files found in the current directory.")
    else:
        plot_3d_subcarriers(csv_files)