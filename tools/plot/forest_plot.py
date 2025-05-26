import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import re
from openood.utils.vis_comm import save_fig_and_close


def extract_ci_values(ci_str):
    """Extract lower and upper CI values in the format '(lower, upper)'."""
    if pd.isna(ci_str):
        return None, None

    # Extract numbers from the string using regex
    matches = re.findall(r'[-+]?\d*\.\d+|\d+', ci_str)
    if len(matches) >= 2:
        return float(matches[0]), float(matches[1])
    return None, None


def plot_forest_chart(csv_file,
                      output_dir,
                      sort_option='ascending',
                      xlim=None):
    """Generate forest plots from a CSV file containing metrics and CI values.

    Args:
        csv_file: Path to the CSV file
        output_dir: Directory to save the output plots
        sort_option: How to sort values - 'ascending', 'descending', or 'none'
        xlim: Upper limit for x-axis. If None, will be calculated automatically
    """
    # Read CSV data
    df = pd.read_csv(csv_file)

    # Determine which metrics to plot
    metrics = [
        col for col in df.columns
        if col != 'Method' and not col.startswith('CI')
    ]

    file_name = os.path.splitext(os.path.basename(csv_file))[0]

    # Process each metric separately
    for metric in metrics:
        # Initialize lists to store data
        methods = []
        metric_values = []
        ci_lowers = []
        ci_uppers = []

        # Process data - expecting alternating rows of values and CIs
        for i in range(0, len(df), 2):
            if i + 1 < len(df) and df.iloc[i + 1, 0] == 'CI':
                method = df.iloc[i, 0]

                # Skip if the metric column doesn't exist or value is NaN
                if metric not in df.columns or pd.isna(df.iloc[i][metric]):
                    continue

                metric_value = float(df.iloc[i][metric])
                ci_str = df.iloc[i + 1][metric]
                ci_lower, ci_upper = extract_ci_values(ci_str)

                if ci_lower is not None and ci_upper is not None:
                    methods.append(method)
                    metric_values.append(metric_value)
                    ci_lowers.append(ci_lower)
                    ci_uppers.append(ci_upper)

        # Skip if no valid data for this metric
        if not methods:
            print(f'No valid data for metric: {metric}. Skipping.')
            continue

        # Convert to dataframe for easier handling
        plot_df = pd.DataFrame({
            'Method': methods,
            'Value': metric_values,
            'CI_Lower': ci_lowers,
            'CI_Upper': ci_uppers
        })

        # Compute error bars
        plot_df['error_lower'] = plot_df['Value'] - plot_df['CI_Lower']
        plot_df['error_upper'] = plot_df['CI_Upper'] - plot_df['Value']

        # Sort by performance for better readability (if requested)
        if sort_option != 'none':
            sort_ascending = (sort_option == 'ascending')
            plot_df = plot_df.sort_values(by='Value', ascending=sort_ascending)
        # If sort_option is 'none', preserve the original order from the CSV

        # Get errors array from the (potentially sorted) dataframe
        errors = [plot_df['error_lower'].values, plot_df['error_upper'].values]

        # Get the number of methods for sizing
        num_methods = len(plot_df)

        # Adjust figure height based on number of methods and desired spacing
        # Increase the multiplier (0.5) to create more space between rows
        row_height = 0.5  # Increase to add more space between rows
        fig_height = max(5, num_methods * row_height)

        # Plot with adjusted height
        plt.figure(figsize=(8, fig_height))

        # Create y positions with increased spacing
        y_positions = np.arange(len(plot_df))

        # Plot using the custom y_positions
        plt.errorbar(plot_df['Value'],
                     y_positions,
                     xerr=errors,
                     fmt='o',
                     capsize=4,
                     markersize=4,
                     ecolor='#888888',
                     elinewidth=1.5)

        # After creating the plot, add background shading for alternating rows
        for i, y in enumerate(y_positions):
            if i % 2 == 0:
                plt.axhspan(y - 0.4, y + 0.4, color='#f5f5f5', zorder=0)

        # Update the method labels on y-axis
        plt.yticks(y_positions, plot_df['Method'])

        # Add more padding to prevent CI labels from being cut off
        # Calculate the space needed for text annotations
        rel_text_padding = 0.02  # Estimate of space needed in data units
        x_min = min(plot_df['CI_Lower'])
        x_max = max(plot_df['CI_Upper'])
        if xlim is not None:
            x_max = xlim * (1 + rel_text_padding)
            x_min = 0 * (1 - rel_text_padding)
        x_range = x_max - x_min
        x_max = x_max + x_range * rel_text_padding
        x_min = x_min - x_range * rel_text_padding

        # Add CI values precisely at the start and end of each bar
        for i, (idx, row) in enumerate(plot_df.iterrows()):
            lower_value = row['CI_Lower']
            upper_value = row['CI_Upper']
            point_value = row['Value']
            lower_text = f'{lower_value:.0f}' \
                if np.abs(lower_value - round(lower_value)) < 0.01 \
                else f'{lower_value:.1f}'
            upper_text = f'{upper_value:.0f}' \
                if np.abs(upper_value - round(upper_value)) < 0.01 \
                else f'{upper_value:.1f}'
            value_text = f'{point_value:.0f}' \
                if np.abs(point_value - round(point_value)) < 0.01 \
                else f'{point_value:.2f}'
            if lower_text.endswith('.0'):
                lower_text = lower_text[:-2]
            if upper_text.endswith('.0'):
                upper_text = upper_text[:-2]
            if value_text.endswith('.00'):
                value_text = value_text[:-3]
            elif '.' in value_text and value_text.endswith(
                    '0'):  # such as 93.80
                value_text = value_text[:-1]

            # Calculate width of CI
            ci_width = row['CI_Upper'] - row['CI_Lower']

            # Calculate relative CI width compared to the full plot range
            plot_range = x_max - x_min
            relative_ci_width = ci_width / plot_range if plot_range > 0 else 0

            # Define thresholds as percentages of the plot range
            narrow_threshold = 0.05  # CI width is 5% of plot range
            very_narrow_threshold = 0.04  # CI width is 1% of plot range
            very_very_narrow_threshold = 0.01  # CI width is 1% of plot range

            # Adjust x_adjust based on relative CI width
            x_adjust = 0 if relative_ci_width > narrow_threshold \
                else plot_range * 0.1
            x_adjust += 0 if relative_ci_width > very_narrow_threshold \
                else plot_range * 0.06
            x_adjust += 0 if relative_ci_width > very_very_narrow_threshold \
                else plot_range * 0.06

            # Use y_positions for annotation placement instead of method names
            if ci_width > 0:
                plt.annotate(lower_text,
                             xy=(row['CI_Lower'], y_positions[i]),
                             xytext=(0 - x_adjust, 3),
                             textcoords='offset points',
                             ha='center',
                             va='bottom',
                             fontsize=8)

                plt.annotate(upper_text,
                             xy=(row['CI_Upper'], y_positions[i]),
                             xytext=(0 + x_adjust, 3),
                             textcoords='offset points',
                             ha='center',
                             va='bottom',
                             fontsize=8)

            # Position metric value below the point
            plt.annotate(value_text,
                         xy=(row['Value'], y_positions[i]),
                         xytext=(0, -6),
                         textcoords='offset points',
                         ha='center',
                         va='top',
                         weight='bold',
                         fontsize=9)

        # Set plot labels and title
        plt.xlabel(f'{metric} (%)', fontweight='bold')
        plt.ylabel('Method', fontweight='bold')
        plt.title(f'{metric} with 95% Confidence Intervals',
                  fontweight='bold',
                  pad=15)

        # Add both horizontal and vertical grid lines
        plt.grid(True, linestyle='--', alpha=0.7)

        # Set x limits to show the full range needed
        plt.xlim(x_min, x_max)
        # Reduce extra space below the bottom row
        plt.ylim(min(y_positions) - row_height, max(y_positions) + row_height)

        # Ensure x-ticks are within the valid range (0-100)
        xticks = plt.xticks()[0]  # Get current ticks
        valid_xticks = [tick for tick in xticks if 0 <= tick <= 100]
        plt.xticks(valid_xticks)

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Save the plot with metric name in the filename (similar to bar_plot)
        safe_metric = metric.replace(' ', '_').replace('/', '_')
        output_path = os.path.join(output_dir,
                                   f'{file_name}_{safe_metric}.svg')
        save_fig_and_close(output_path)

        print(f'Created forest plot for {metric} at {output_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Plot forest charts from CSV data')
    parser.add_argument('csv_file', help='Path to the CSV file')
    parser.add_argument('output_dir',
                        help='Directory to save the output plot images')
    parser.add_argument('--sort',
                        choices=['none', 'ascending', 'descending'],
                        default='none',
                        help='Sort order (default: none)')
    parser.add_argument('--xlim',
                        type=float,
                        default=None,
                        help='Upper limit for x-axis. Default: auto')

    args = parser.parse_args()

    plot_forest_chart(args.csv_file, args.output_dir, args.sort, args.xlim)
