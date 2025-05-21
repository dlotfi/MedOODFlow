import argparse
import pandas as pd
import matplotlib.pyplot as plt
from openood.utils.vis_comm import save_fig_and_close, MARKERS


def plot_csv_data(file_path, title, output_file):
    # Read the CSV file
    data = pd.read_csv(file_path)

    # Extract the column names
    columns = data.columns

    # X-axis values (first column as strings)
    x_labels = data[columns[0]].astype(str)
    # Use the index for evenly spaced x-ticks
    x = range(len(x_labels))

    # Plot each column from the second onward
    for i, column in enumerate(columns[1:], start=1):
        plt.plot(x,
                 data[column],
                 marker=MARKERS[i % len(MARKERS)],
                 label='All' if i == 1 else column)

    # Set plot title and axis labels
    plt.title(title)  # Title of the plot
    plt.xlabel(columns[0])  # Title of the first column as x-axis label
    plt.ylabel(columns[1])  # Title of the second column as y-axis label

    # Set x-axis ticks to evenly spaced values with labels from column 1
    plt.xticks(x, x_labels)

    # Add legend outside the plot
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)

    # Add grid for better visibility
    plt.grid(True)

    # Adjust the layout to make room for the legend and avoid overlaps
    plt.tight_layout(rect=(0, 0.1, 1, 1))

    # Save the plot to the specified output file
    save_fig_and_close(output_file)


if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Plot a CSV file and save the output as an image file.')
    parser.add_argument('file_path', type=str, help='Path to the CSV file.')
    parser.add_argument('title', type=str, help='Title of the plot.')
    parser.add_argument('output_file',
                        type=str,
                        help='Path to save the output plot image.')

    args = parser.parse_args()

    # Call the plotting function with the provided arguments
    plot_csv_data(args.file_path, args.title, args.output_file)
