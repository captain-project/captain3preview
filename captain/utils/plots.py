import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from PIL import Image

logger = logging.getLogger(__name__)


def plot_grid(
    data,
    mask=None,
    title=None,
    outfile=None,
    cmap="YlGnBu",
    background="lightgrey",
    zero_color="white",
    rescale_figure: float = 1.0,
    dpi: int = 100,
    figsize=(5, 6),
    vmin=None,
    vmax=None,
):
    # 1. Prepare Data
    plot_data = np.array(data).copy()
    if mask is not None:
        plot_data[~mask] = np.nan

    # 2. Setup Figure
    fig, ax = plt.subplots(
        figsize=(figsize[0] * rescale_figure, figsize[1] * rescale_figure)
    )

    # Set the background color (for NAs)
    ax.set_facecolor(background)

    # 3. LAYER 1: The Zero Cells (No colorbar)
    zero_mask = (plot_data != 0) | np.isnan(plot_data)
    if not np.all(zero_mask):
        sns.heatmap(
            np.zeros_like(plot_data),
            mask=zero_mask,
            cmap=[zero_color],
            cbar=False,  # Keep this False
            xticklabels=False,
            yticklabels=False,
            ax=ax,
        )

    # 4. LAYER 2: The Actual Data with Horizontal Colorbar
    data_mask = np.isnan(plot_data) | (plot_data == 0)

    if not np.all(data_mask):
        sns.heatmap(
            plot_data,
            mask=data_mask,
            cmap=cmap,
            xticklabels=False,
            yticklabels=False,
            ax=ax,
            # Configure the colorbar position and orientation
            cbar_kws={
                "orientation": "horizontal",
                "pad": 0.08,  # Space between plot and colorbar
                "shrink": 0.8,  # Makes the bar slightly shorter than the plot width
            },
            vmin=vmin,
            vmax=vmax,
        )

    if title:
        ax.set_title(title)

    # 5. Add frame/spines
    for _, spine in ax.spines.items():
        spine.set_visible(True)
        spine.set_linewidth(0.5)
        spine.set_color("black")

    plt.tight_layout()

    if outfile is None:
        plt.show()
    else:
        plt.savefig(outfile, dpi=dpi)  # , bbox_inches='tight', pad_inches=0.01)

    plt.close(fig)


def create_gif(png_files, duration_ms=100, rm_png=False):
    """
    Combines all PNG files in a folder into a single GIF.

    Args:
        image_folder (str): The path to the folder containing the PNG files.
        gif_name (str): The desired name for the output GIF file.
        duration_ms (int): The duration of each frame in milliseconds.
    """
    # Create a list to store image objects
    frames = []

    # Open and append each image to the frames list
    for file_name in png_files:
        frames.append(Image.open(file_name))

    # Save the frames as an animated GIF
    # The first frame is used to save the sequence

    frames[0].save(
        os.path.join(png_files[0].replace(".png", ".gif")),
        save_all=True,
        append_images=frames[1:],
        optimize=False,
        duration=duration_ms,
        loop=0,
    )
    if rm_png:
        _ = [os.remove(f) for f in png_files]


def plot_extinction_risk(
    data,
    labels,
    title="Conservation Status Distribution",
    outfile=None,
    dpi=100,
    ymax=None,
):
    """
    Plots a bar chart of conservation status counts.

    Args:
        data: NumPy array of integers (0-4).
        labels: List of 5 strings for the X-axis (e.g., ['LC', 'NT', 'VU', 'EN', 'CR']).
        title: Title of the plot.
        outfile: Path to save the PNG. If None, it calls plt.show().
    """

    if torch.is_tensor(data):
        data = data.detach().cpu().numpy()

    counts = np.bincount(data.astype(int), minlength=len(labels))

    plt.figure(figsize=(8, 5), dpi=dpi)

    cmap = plt.get_cmap("RdYlGn")
    colors = cmap(np.linspace(1, 0, len(labels)))

    bars = plt.bar(labels, counts, color=colors, edgecolor="black", linewidth=0.8)

    # Calculate unified Y-limit with a 15% headroom for the text labels
    y_limit = (
        ymax * 1.15
        if ymax is not None
        else (max(counts) if len(counts) > 0 else 1) * 1.15
    )

    plt.ylim(0, y_limit)

    plt.title(title, fontsize=14, fontweight="bold", pad=15)
    plt.ylabel("Number of Species", fontsize=12)
    plt.xlabel("Status", fontsize=12)

    # Add count labels on top of each bar using a dynamic vertical offset
    text_offset = y_limit * 0.02
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + text_offset,
            f"{int(height)}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    plt.grid(axis="y", linestyle="--", alpha=0.3)
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)

    plt.tight_layout()

    if outfile:
        plt.savefig(outfile, bbox_inches="tight", dpi=dpi)
        logger.info("Plot saved to %s", outfile)
    else:
        plt.show()

    plt.close()


# --- Example Usage ---
# status_labels = ["Least Concern", "Near Threatened", "Vulnerable", "Endangered", "Critically Endangered"]
# plot_conservation_distribution(conservation_status, status_labels, outfile="status_plot.png")


def plot_rl_rewards(
    file_path,
    start_span=None,
    end_span=None,
    title="RL training rewards",
    outfile=None,
    dpi=300,
    reward_col="reward",
):
    df = pd.read_csv(file_path, sep="\t")
    rewards = df[reward_col].values
    epochs = np.arange(len(rewards))

    # 1. Define your span range
    # Start with a small span (very reactive) and grow to a larger one (very smooth).
    # Defaults scale with the number of epochs so the EMA can actually catch up
    # to the raw reward by the end of the run: fixed defaults tuned for long
    # (hundreds+ epoch) runs leave alpha too small to track short runs (e.g. a
    # 20-epoch smoke test), making the smoothed line trail well below the raw
    # reward instead of converging to it.
    if start_span is None:
        start_span = max(2, min(30, len(rewards) // 4))
    if end_span is None:
        end_span = max(start_span + 1, len(rewards))

    # 2. Create an array of alphas that decrease over time
    # (Since larger span = smaller alpha = more smoothing)
    spans = np.exp(np.linspace(np.log(start_span), np.log(end_span), len(rewards)))
    alphas = 2 / (spans + 1)

    # 3. Compute the Dynamic EMA
    # We have to do this in a loop because 'span' in .ewm() doesn't accept an array
    smoothed = np.zeros(len(rewards))
    smoothed[0] = rewards[0]
    for t in range(1, len(rewards)):
        # Formula: y_t = (1 - alpha)*y_{t-1} + alpha*x_t
        smoothed[t] = (1 - alphas[t]) * smoothed[t - 1] + alphas[t] * rewards[t]

    # 4. Plotting
    plt.figure(figsize=(8, 4.5))
    plt.scatter(epochs, rewards, color="tab:blue", alpha=0.15, s=8)
    plt.plot(
        epochs, smoothed, color="crimson", linewidth=2.5, label="Dynamic Span Trend"
    )

    plt.title(title, fontsize=14)
    plt.xlabel("Epoch")
    plt.ylabel("Reward")
    plt.gca().spines[["top", "right"]].set_visible(False)
    plt.tight_layout()

    if outfile:
        plt.savefig(outfile, bbox_inches="tight", dpi=dpi)
        logger.info("Plot saved to %s", outfile)
    else:
        plt.show()

    plt.close()
