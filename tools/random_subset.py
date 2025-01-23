import random
import argparse
from collections import defaultdict


def select_with_distribution(input_path, percentage, output_path, seed):
    """Randomly selects a percentage of lines from an input file while
    maintaining label distribution and preserving the relative order of lines.

    Parameters:
        input_path (str): Path to the input file.
        percentage (float): Percentage of lines to select (e.g., 50 for 50%).
        output_path (str): Path to the output file.
        seed (int): Seed for randomization to ensure reproducibility.
    """
    # Set the random seed for reproducibility
    random.seed(seed)

    # Read the input file and organize indices by label
    indices_by_label = defaultdict(list)
    with open(input_path, 'r') as f:
        lines = f.readlines()
        for idx, line in enumerate(lines):
            # Extract label from each line and group indices by label
            label = line.strip().rsplit(' ', 1)[-1]
            indices_by_label[label].append(idx)

    # Determine the number of samples to select for each label
    selected_indices = []
    for label, indices in indices_by_label.items():
        # At least select one if non-empty
        num_to_select = max(1, int(len(indices) * (percentage / 100)))
        selected_indices.extend(random.sample(indices, num_to_select))

    # Sort the selected indices to preserve the original order of lines
    selected_indices.sort()

    # Write the selected data to the output file
    with open(output_path, 'w') as f:
        for idx in selected_indices:
            f.write(lines[idx])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Randomly select a percentage of lines from a file while '
        'maintaining label distribution.')
    parser.add_argument('input_path', type=str, help='Path to the input file.')
    parser.add_argument('percentage',
                        type=float,
                        help='Percentage of lines to select.')
    parser.add_argument('output_path',
                        type=str,
                        help='Path to the output file.')
    parser.add_argument('seed', type=int, help='Seed for reproducibility.')

    # Parse arguments
    args = parser.parse_args()

    # Run the selection function
    select_with_distribution(args.input_path, args.percentage,
                             args.output_path, args.seed)
