#!/usr/bin/env python
"""Regenerate the tutorials/toy_example plots from already-trained models.

Loads the trained weights already saved under data/<res_name>_w.../ (does
NOT retrain), runs one episode, and (re)writes:
- images/priority_<res_name>.png   (protection matrix with region outlines)
- images/ext_risk_future_<res_name>.png
- images/ext_risk_future_unprotected.png (global mode only)

The reward-over-training plot is NOT regenerated here — copy
data/<res_name>_w.../reward_over_training.png directly, since that's already
produced by train_policy.py itself via captain.utils.plots.plot_rl_rewards.

Usage:
    uv run python scripts/make_tutorial_plots.py global
    uv run python scripts/make_tutorial_plots.py regions
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
IMAGES = REPO / "tutorials" / "toy_example" / "images"

sys.path.insert(0, str(REPO / "examples"))

N_ROWS, N_COLS = 3, 3  # must match scripts/create_toy_regions.py

mask = np.load(REPO / "data" / "env_layers" / "area_mask.npy")

# Recompute the same straight-line region boundaries create_toy_regions.py
# uses, so the overlay always matches regions.tif without needing to load
# and interpolate the raster itself.
valid_rows = np.where(mask.sum(axis=1) > 0)[0]
valid_cols = np.where(mask.sum(axis=0) > 0)[0]
row_groups = np.array_split(np.arange(valid_rows.min(), valid_rows.max() + 1), N_ROWS)
col_groups = np.array_split(np.arange(valid_cols.min(), valid_cols.max() + 1), N_COLS)
ROW_BOUNDARIES = [g[-1] + 0.5 for g in row_groups[:-1]]
COL_BOUNDARIES = [g[-1] + 0.5 for g in col_groups[:-1]]
ROW_EXTENT = (valid_rows.min() - 0.5, valid_rows.max() + 0.5)
COL_EXTENT = (valid_cols.min() - 0.5, valid_cols.max() + 0.5)


def add_region_outlines(ax):
    for rb in ROW_BOUNDARIES:
        ax.plot(COL_EXTENT, [rb, rb], color="black", linewidth=0.9, zorder=5)
    for cb in COL_BOUNDARIES:
        ax.plot([cb, cb], ROW_EXTENT, color="black", linewidth=0.9, zorder=5)


def plot_priority_with_regions(protection_grid, title, outfile):
    plot_data = np.array(protection_grid).copy()
    plot_data[mask == 0] = np.nan

    fig, ax = plt.subplots(figsize=(6, 8))
    ax.set_facecolor("lightgrey")

    white_layer = np.where(np.isfinite(plot_data) & (plot_data == 0), 1.0, np.nan)
    ax.imshow(white_layer, cmap="gray", vmin=0, vmax=1, interpolation="none", origin="upper")

    data_layer = np.ma.masked_where(~np.isfinite(plot_data) | (plot_data == 0), plot_data)
    im = ax.imshow(data_layer, cmap="YlGnBu", vmin=0.9, vmax=1.1, interpolation="none", origin="upper")

    add_region_outlines(ax)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.5)
        spine.set_color("black")
    fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.05, shrink=0.8)
    plt.tight_layout()
    plt.savefig(outfile, dpi=300)
    plt.close(fig)
    print("saved", outfile)


def make_all(use_regional: bool, res_name: str):
    import train_policy as tp

    tp.USE_REGIONAL_AGENTS = use_regional
    tp.REGION_ID = None
    model_dir = tp.DATA_DIR / (
        res_name + "_w" + "_w".join(str(r) for r in tp.REWARD_WEIGHTS) + f"_p{tp.TARGET_PROTECTED_CELLS_FRACTION}"
    )
    weights = np.load(model_dir / "trained_weights.npy")

    ep = tp.create_episode_runner()
    ep.policy.set_flat_weights(weights)
    ep.rewards = tp.cn.NoRewards()
    ep.run_episode(weights)
    protection_grid = ep.env.protection_matrix.reconstruct_grid[0]

    label = "Regional" if use_regional else "Global"
    plot_priority_with_regions(
        protection_grid, "protection matrix (region outlines)", IMAGES / f"priority_{res_name}.png"
    )

    # future extinction-risk plot (with this policy's protection)
    tp.cn.plots.plot_extinction_risk(
        ep.env.current_ext_risk,
        labels=["LC", "NT", "VU", "EN", "CR"],
        outfile=IMAGES / f"ext_risk_future_{res_name}",
        title=f"Future extinction risk ({label.lower()} agent)",
        dpi=200,
    )

    # no-protection baseline (global run only; RegionalPolicyNetwork isn't
    # compatible with NoBudgetManager, matching run_inference.py's guard)
    if not use_regional:
        ep_noprotect = tp.cn.EpisodeRunner(
            env=ep.env,
            feature_extractor=ep.feature_extractor,
            policy_network=ep.policy,
            rewards=tp.cn.NoRewards(),
            n_steps=tp.N_TIME_STEPS,
            budget_manager=tp.cn.NoBudgetManager(),
        )
        ep_noprotect.run_episode(weights)
        tp.cn.plots.plot_extinction_risk(
            ep.env.current_ext_risk,
            labels=["LC", "NT", "VU", "EN", "CR"],
            outfile=IMAGES / "ext_risk_future_unprotected",
            title="Future extinction risk (no protection)",
            dpi=200,
        )

    print(
        f"NOTE: reward_{res_name}.png was not touched — copy it directly from "
        f"{model_dir / 'reward_over_training.png'}"
    )


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "global":
        make_all(False, "global")
    elif mode == "regions":
        make_all(True, "regions")
    else:
        raise SystemExit(f"Unknown mode {mode!r}: expected 'global' or 'regions'")
