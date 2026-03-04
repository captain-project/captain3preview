#!/usr/bin/env python
"""Example: Train a conservation policy with Evolution Strategies.

This script demonstrates how to:
1. Load real spatial data (habitat suitability maps, disturbance, costs, ...)
2. Plot the data and their evolution through time

Requirements:
- Example data in DATA_DIR (see below)
- Species trait CSV file

"""

import warnings

# Filter out the specific PyTorch Sparse CSR beta warning
warnings.filterwarnings("ignore", message="Sparse CSR tensor support is in beta state")
import os
from pathlib import Path
import numpy as np
import captain as cn

SEED = None
# =============================================================================
# Configuration
# =============================================================================

# Data paths - UPDATE THESE to point to your data
DATA_DIR = Path("/path/to/your/captain3data")  # <-- Change this!

PRESENT_SDMS_DIR = "present_sdms"
FUTURE_SDMS_DIR = "future_sdms"
SPECIES_TRAIT_FILE = "species_tbl.csv"
DISTURBANCE_FILE = "env_layers/area_swept_disturbance.tif"
FUTURE_DISTURBANCE_FILE = "env_layers/future_area_swept_disturbance.tif"
COST_FILE = "env_layers/cost.tif"
FUTURE_COST_FILE = "env_layers/future_cost.tif"
DATA_MASK = "env_layers/area_mask.npy"
results_dir = "plots"
os.makedirs(DATA_DIR / results_dir, exist_ok=True)

# Time duration of each episode
N_TIME_STEPS = 50  

# Minimum habitat suitability threshold
MIN_HABITAT_SUITABILITY = 0.05  # can be an array (per-species values)

# Output
RES_DIR = DATA_DIR / results_dir

# =============================================================================
# Episode Setup Function
# =============================================================================
# Check data directory
if not DATA_DIR.exists() or str(DATA_DIR) == "/path/to/your/data":
    print("\nERROR: Please update DATA_DIR in this script to point to your data.")
    print("       See the example data repository for the expected format.")
    raise FileNotFoundError

# Load present and future species distribution maps
mask, _ = cn.data_loader.load_map(DATA_DIR / DATA_MASK)

sdm = cn.load_spatial_data_from_dir(
    dir=DATA_DIR / PRESENT_SDMS_DIR,
    future_dir=DATA_DIR / FUTURE_SDMS_DIR,
    mask=mask,
    lower_bound=0,
    upper_bound=1,
    n_time_steps=N_TIME_STEPS,
    min_threshold=MIN_HABITAT_SUITABILITY,
)

# species index (list stored in sdm.names)
species_name = "Agarophyton.chilense"
species_i = sdm.names.index(species_name)
cn.data.spatial_data.plot_data_evolution(
    sdm,
    n_steps=N_TIME_STEPS,
    skip=1,
    title=f"{sdm.names[species_i]}",
    indx=species_i,
    outfile=RES_DIR / f"{sdm.names[species_i]}",
    vmin=0,
    vmax=1,
)

cn.plots.plot_grid(
    np.sum(sdm.reconstruct_grid > 0.75, axis=0),
    title="Species richness (present habitat suitability)",
    outfile=RES_DIR / "Species_richness_present",
    vmin=0,
    vmax=175,
    cmap="Blues",
)

sdm.update(50)
cn.plots.plot_grid(
    np.sum(sdm.reconstruct_grid > 0.75, axis=0),
    title="Species richness (future habitat suitability)",
    outfile=RES_DIR / "Species_richness_future",
    vmin=0,
    vmax=175,
    cmap="Blues",
)

# Load disturbance layer with predicted future change
disturbance = cn.load_spatial_data(
    file=DATA_DIR / DISTURBANCE_FILE,
    future_file=DATA_DIR / FUTURE_DISTURBANCE_FILE,
    mask=mask,
    lower_bound=0,
    upper_bound=1,
    n_time_steps=N_TIME_STEPS,
)

cn.data.spatial_data.plot_data_evolution(
    disturbance,
    n_steps=N_TIME_STEPS,
    skip=1,
    title="Disturbance",
    outfile=RES_DIR / "disturbance",
    vmin=0,
    vmax=1,
)

# Load costs with predicted future change
costs = cn.load_spatial_data(
    file=DATA_DIR / COST_FILE,
    future_file=DATA_DIR / FUTURE_COST_FILE,
    mask=mask,
    lower_bound=0,
    upper_bound=1,
    n_time_steps=N_TIME_STEPS,
)

cn.data.spatial_data.plot_data_evolution(
    costs,
    n_steps=50,
    skip=1,
    title="Costs",
    outfile=RES_DIR / "costs",
    vmin=0,
    vmax=1,
)

# Load species traits
# simple imputation of missing data (could be replaced e.g. RF imputation)
traits = cn.data_loader.load_trait_table(
    DATA_DIR / SPECIES_TRAIT_FILE,
    species_list=sdm.names,
    ref_column="species",
    fill_gaps=True,
)

conservation_status = traits["conservation_status"].to_numpy(copy=True) - 1

# Initial extinction risk from conservation status
ext_risk = cn.ExtinctionRisk(
    init_status=conservation_status,
    n_classes=5,
    alpha=0.5,
)
cn.plots.plot_extinction_risk(
    ext_risk.init_status,
    labels=["LC", "NT", "VU", "EN", "CR"],
    outfile=RES_DIR / "Extinction_risk",
    dpi=200,
)
