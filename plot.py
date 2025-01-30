from datetime import datetime

import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
import matplotlib.dates as mdates
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from parse import get_amplitudes, get_phases


class PlotCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None, width=6, height=3, dpi=150):
        self.fig = Figure(
            figsize=(width, height),
            dpi=dpi,
            facecolor="#1E1E1E",
            constrained_layout=True,
        )
        self.fig.set_tight_layout(dict(pad=0, h_pad=0, w_pad=0))

        super().__init__(self.fig)
        self.setParent(parent)
        self.setStyleSheet("background-color: #1E1E1E;")

        self.ax = self.fig.add_subplot(111)
        self.ax.set_position([0, 0, 1, 1])  # Make axes fill figure
        self._setup_plot_style()
        self._draw_placeholder()

    def _setup_plot_style(self):
        self.ax.set_facecolor("#1E1E1E")
        self.ax.grid(False)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        for spine in self.ax.spines.values():
            spine.set_visible(False)

    def _draw_placeholder(self):
        self.ax.clear()
        self._setup_plot_style()
        self.ax.text(
            0.5,
            0.5,
            "No data loaded",
            ha="center",
            va="center",
            color="#8B8B8B",
            fontsize=10,
            transform=self.ax.transAxes,
        )
        self.draw()

    def update_plot(self, sensor_data):
        self.ax.clear()
        self._setup_plot_style()

        if sensor_data is not None and len(sensor_data) > 0:
            indices = np.arange(len(sensor_data[0]))
            average_readings = np.mean(sensor_data, axis=0)

            # Plot average line
            self.ax.plot(indices, average_readings, color="#00FF9F", linewidth=2)

            # Add min-max range
            self.ax.fill_between(
                indices,
                np.min(sensor_data, axis=0),
                np.max(sensor_data, axis=0),
                color="#00FF9F",
                alpha=0.1,
            )

            self.ax.spines["bottom"].set_visible(True)
            self.ax.spines["bottom"].set_color("#333333")
            self.ax.set_xticks(np.linspace(0, len(indices) - 1, 5, dtype=int))
            self.ax.tick_params(colors="#8B8B8B", labelsize=8)
            self.fig.subplots_adjust(left=0, right=1, top=1, bottom=0.2)
        else:
            self._draw_placeholder()
            self.fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

        self.draw()


def plot_sensor_data(
    sensor_data, timestamps, plot_type="average", minimal=False, alpha=0.1
):
    """
    Plot sensor data with different visualization options.

    Parameters:
    sensor_data (np.array): Array of sensor readings
    timestamps (np.array): Array of Unix timestamps
    plot_type (str): Type of plot ('average', 'overlay', or 'dashboard')
    alpha (float): Transparency for overlaid plots (0-1)
    minimal (bool): If True, creates a minimal version suitable for GUI integration
    """
    plt.style.use("dark_background")

    dates = [datetime.fromtimestamp(ts) for ts in timestamps]
    indices = np.arange(len(dates))

    if plot_type == "average":
        _plot_average(sensor_data, dates, indices, minimal)
    elif plot_type == "overlay":
        _plot_overlay(sensor_data, dates, indices, alpha)
    elif plot_type == "dashboard":
        _plot_dashboard(sensor_data, dates, indices)
    else:
        raise ValueError("plot_type must be 'average', 'overlay', or 'dashboard'")

    plt.show()


def _format_xaxis_with_indices(ax, dates, indices, minimal=False):
    axis_color = "#8B8B8B"
    spine_color = "#333333"
    if minimal:
        # For minimal view, only show indices
        ax.set_xticks(np.linspace(0, len(indices) - 1, 5, dtype=int))
        ax.tick_params(colors=axis_color, labelsize=8)

        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.spines["bottom"].set_visible(True)
        ax.spines["bottom"].set_color(spine_color)

        return None
    else:
        # Regular view with both dates and indices
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M"))
        plt.setp(
            ax.xaxis.get_majorticklabels(), rotation=45, ha="right", color=axis_color
        )

        ax2 = ax.twiny()
        ax2.set_xlim(ax.get_xlim())

        num_ticks = 10
        tick_indices = np.linspace(0, len(indices) - 1, num_ticks, dtype=int)
        ax2.set_xticks(tick_indices)
        ax2.set_xticklabels([f"[{i}]" for i in tick_indices], color=axis_color)
        ax2.set_xlabel("Array Index", color=axis_color)

        for spine in ax.spines.values():
            spine.set_color(spine_color)
        for spine in ax2.spines.values():
            spine.set_color(spine_color)

        return ax2


def _plot_average(sensor_data, dates, indices, minimal=False):
    figsize = (6, 3) if minimal else (12, 6)
    fig, ax = plt.subplots(figsize=figsize)
    facecolor = "#1E1E1E"
    label_color = "#8B8B8B"
    plot_color = "#00FF9F"

    fig.set_facecolor(facecolor)
    ax.set_facecolor(facecolor)

    # Plot average line and fill
    average_readings = np.mean(sensor_data, axis=0)
    _ = ax.plot(
        indices,
        average_readings,
        color=plot_color,
        linewidth=2,
        label="Average Reading",
    )[0]

    ax.fill_between(
        indices,
        np.min(sensor_data, axis=0),
        np.max(sensor_data, axis=0),
        color=plot_color,
        alpha=0.1,
        label="Min-Max Range",
    )

    if not minimal:
        # Full view styling
        ax.set_title(
            "Average Sensor Reading Over Time", color="white", pad=20, fontsize=12
        )
        ax.set_xlabel("Time", color=label_color, labelpad=10)
        ax.set_ylabel("Reading Value", color=label_color, labelpad=10)
        legend = ax.legend(frameon=False)
        plt.setp(legend.get_texts(), color=label_color)
    else:
        # Minimal view styling
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_title("")
        ax.yaxis.set_visible(False)
        ax.spines["left"].set_visible(False)

    ax.grid(False)
    _format_xaxis_with_indices(ax, dates, indices, minimal)
    plt.tight_layout()


def _plot_overlay(sensor_data, dates, indices, alpha):
    _, ax = plt.subplots(figsize=(12, 6))

    for i in range(sensor_data.shape[0]):
        ax.plot(indices, sensor_data[i, :], alpha=alpha, label=f"Sensor {i+1}")

    ax.set_title("All Sensor Readings")
    ax.set_xlabel("Time")
    ax.set_ylabel("Reading Value")
    ax.grid(True)

    # Add legend if there aren't too many sensors
    if sensor_data.shape[0] <= 10:
        ax.legend()

    _format_xaxis_with_indices(ax, dates, indices)
    plt.tight_layout()


def _plot_dashboard(sensor_data, dates, indices):
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(3, 2, figure=fig)

    # 1. Average reading with min-max range
    ax1 = fig.add_subplot(gs[0, :])
    average_readings = np.mean(sensor_data, axis=0)
    ax1.plot(indices, average_readings, label="Average")
    ax1.fill_between(
        indices,
        np.min(sensor_data, axis=0),
        np.max(sensor_data, axis=0),
        alpha=0.2,
        label="Min-Max Range",
    )
    ax1.set_title("Average Reading with Range")
    ax1.grid(True)
    ax1.legend()
    _format_xaxis_with_indices(ax1, dates, indices)

    # 2. Heatmap of all sensors
    ax2 = fig.add_subplot(gs[1, :])
    im = ax2.imshow(sensor_data, aspect="auto", cmap="viridis")
    ax2.set_title("Sensor Heatmap")
    plt.colorbar(im, ax=ax2)

    # Add time ticks to heatmap
    num_ticks = 10
    tick_positions = np.linspace(0, sensor_data.shape[1] - 1, num_ticks, dtype=int)
    tick_labels = [dates[i].strftime("%Y-%m-%d\n%H:%M") for i in tick_positions]
    ax2.set_xticks(tick_positions)
    ax2.set_xticklabels(tick_labels, rotation=45)
    ax2.set_xlabel("Time [Index]")
    ax2.set_ylabel("Sensor Number")

    # Add secondary axis for indices
    ax2_top = ax2.twiny()
    ax2_top.set_xlim(ax2.get_xlim())
    ax2_top.set_xticks(tick_positions)
    ax2_top.set_xticklabels([f"[{i}]" for i in tick_positions])

    # 3. Distribution of readings
    ax3 = fig.add_subplot(gs[2, 0])
    ax3.hist(sensor_data.flatten(), bins=50)
    ax3.set_title("Distribution of Readings")
    ax3.grid(True)
    ax3.set_xlabel("Reading Value")
    ax3.set_ylabel("Frequency")

    # 4. Box plot of sensors
    ax4 = fig.add_subplot(gs[2, 1])
    ax4.boxplot([sensor_data[i, :] for i in range(min(10, sensor_data.shape[0]))])
    ax4.set_title("Sensor Statistics (Top 10 Sensors)")
    ax4.set_xlabel("Sensor Number")
    ax4.set_ylabel("Reading Value")
    ax4.grid(True)

    plt.tight_layout()


def main():
    file = "data/test_tstream_1736805171_001.npy"
    sensor_data, timestamps = get_amplitudes(np.load(file))

    plot_sensor_data(sensor_data, timestamps, plot_type="average", minimal=True)
    # plot_sensor_data(sensor_data, timestamps, plot_type='overlay', alpha=0.1)
    # plot_sensor_data(sensor_data, timestamps, plot_type='dashboard')


if __name__ == "__main__":
    main()
