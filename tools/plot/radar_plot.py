import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection
from matplotlib.spines import Spine
from matplotlib.path import Path
import matplotlib.patches as mpatches
from openood.utils.vis_comm import save_fig_and_close


def radar_factory(num_vars, frame='polygon'):
    """Create a radar chart with `num_vars` axes and specified frame shape."""
    theta = np.linspace(0, 2 * np.pi, num_vars, endpoint=False)

    class RadarAxes(PolarAxes):
        name = 'radar'
        RESOLUTION = 1

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.set_theta_zero_location('N')
            self.set_theta_direction(-1)

        def fill(self, *args, closed=True, **kwargs):
            return super().fill(*args, closed=closed, **kwargs)

        def plot(self, *args, **kwargs):
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)
            return lines

        def _close_line(self, line):
            x, y = line.get_data()
            if x[0] != x[-1]:
                x = np.concatenate((x, [x[0]]))
                y = np.concatenate((y, [y[0]]))
                line.set_data(x, y)

        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels)

        def _gen_axes_patch(self):
            if frame == 'circle':
                return super()._gen_axes_patch()
            verts = unit_poly_verts(theta)
            return mpatches.Polygon(verts, closed=True, edgecolor='black')

        def _gen_axes_spines(self):
            if frame == 'circle':
                return super()._gen_axes_spines()
            verts = unit_poly_verts(theta)
            path = Path(verts + [verts[0]])
            spine = Spine(self, 'circle', path)
            spine.set_transform(self.transAxes)
            return {'polar': spine}

    def unit_poly_verts(theta):
        """Return vertices of polygon centered at (0.5, 0.5), radius=0.5."""
        x0, y0, r = [0.5] * 3
        return [(r * np.cos(t + np.pi / 2) + x0,
                 r * np.sin(t + np.pi / 2) + y0) for t in theta]

    register_projection(RadarAxes)
    return theta


def plot_radar_chart(csv_file,
                     title,
                     output_file,
                     shape='polygon',
                     invert=False):
    df = pd.read_csv(csv_file)
    labels = df.columns[1:]
    num_vars = len(labels)
    theta = radar_factory(num_vars, frame=shape)

    data = df.iloc[:, 1:].values
    methods = df.iloc[:, 0].tolist()

    # Invert values if requested (100 - value)
    if invert:
        data = 100 - data

    fig, ax = plt.subplots(figsize=(10, 8),
                           subplot_kw=dict(projection='radar'))

    # Set grid lines and labels
    if invert:
        # For inverted: 100 is at center, 0 at edge
        grid_values = [80, 60, 40, 20]
        grid_labels = ['20', '40', '60', '80']
    else:
        # Normal: 0 at center, 100 at edge
        grid_values = [20, 40, 60, 80]
        grid_labels = ['20', '40', '60', '80']

    ax.set_rgrids(grid_values, labels=grid_labels, angle=0)
    ax.set_ylim(0, 100)

    # Make labels bigger and position them outside the radar area
    ax.set_thetagrids(np.degrees(theta), labels, fontsize=12)
    for label, angle in zip(ax.get_xticklabels(), np.degrees(theta)):
        if angle == 0:
            continue
        elif angle < 180:
            label.set_horizontalalignment('left')
        else:
            label.set_horizontalalignment('right')

    for i, (method, row) in enumerate(zip(methods, data)):
        values = row.tolist()
        ax.plot(theta, values, label=method, linewidth=1.5, alpha=0.7)
        # Uncomment to fill
        # ax.fill(theta, values, alpha=0.1)

    ax.grid(alpha=0.8, linestyle='--')

    # Move legend to the bottom
    plt.legend(loc='upper center',
               bbox_to_anchor=(0.5, -0.01),
               ncol=3,
               fontsize=12)
    plt.title(title, size=16, pad=30)
    # Adjust layout to make room for the legend
    plt.tight_layout(rect=[0, 0.01, 1, 0.95])

    # Save the plot to the specified output file
    save_fig_and_close(output_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Plot radar chart from CSV data')
    parser.add_argument('csv_file', help='Path to the CSV file')
    parser.add_argument('title', help='Title of the plot')
    parser.add_argument('output_file',
                        help='Path to save the output plot image')
    parser.add_argument('--shape',
                        choices=['polygon', 'circle'],
                        default='polygon',
                        help='Shape of radar chart grid (default: polygon)')
    parser.add_argument('--invert',
                        action='store_true',
                        help='Invert values and reverse grid labels')
    args = parser.parse_args()
    plot_radar_chart(args.csv_file, args.title, args.output_file, args.shape,
                     args.invert)
