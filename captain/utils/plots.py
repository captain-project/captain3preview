import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from PIL import Image


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
        data, labels, title="Conservation Status Distribution", outfile=None, dpi=100
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
        # .detach() removes it from the computational graph
        # .cpu() ensures it is on system memory, not GPU
        # .numpy() converts it
        data = data.detach().cpu().numpy()

    # 1. Calculate counts for exactly 5 categories (0 through 4)
    # minlength ensures that if a category has 0 counts, it still appears in the array
    counts = np.bincount(data.astype(int), minlength=len(labels))

    # 2. Create the figure
    plt.figure(figsize=(8, 5), dpi=dpi)

    # 3. Define the Green-to-Red Colormap
    # 'RdYlGn' goes from Red (0.0) to Green (1.0).
    # Since we want Status 0 (Safe) to be Green, we sample from 1.0 down to 0.0.
    cmap = plt.get_cmap("RdYlGn")
    colors = cmap(np.linspace(1, 0, len(labels)))

    # 4. Create the Bar Plot
    bars = plt.bar(labels, counts, color=colors, edgecolor="black", linewidth=0.8)

    # 5. Aesthetics
    plt.title(title, fontsize=14, fontweight="bold", pad=15)
    plt.ylabel("Number of Species", fontsize=12)
    plt.xlabel("Status", fontsize=12)

    # Add count labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.1,
            f"{int(height)}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    # Clean up layout
    plt.grid(axis="y", linestyle="--", alpha=0.3)
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)

    plt.tight_layout()

    if outfile:
        plt.savefig(outfile, bbox_inches="tight", dpi=dpi)
        print(f"Plot saved to {outfile}")
    else:
        plt.show()

    plt.close()

# --- Example Usage ---
# status_labels = ["Least Concern", "Near Threatened", "Vulnerable", "Endangered", "Critically Endangered"]
# plot_conservation_distribution(conservation_status, status_labels, outfile="status_plot.png")
