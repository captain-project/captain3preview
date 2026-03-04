#!/usr/bin/env python
"""Example: Train a conservation policy with Evolution Strategies.

This script demonstrates how to:
1. Load real spatial data (habitat suitability maps, disturbance, costs, ...)
2. Set up an episode runner
3. Load a trained model
4. Run episode and plot results

Requirements:
- Example data in DATA_DIR (see below)
- Species trait CSV file
- Trained model (provided)

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
TRAINED_MODEL = Path("/path/to/your/trained_model/trained_weights.npy")  # <-- Change this!

PRESENT_SDMS_DIR = "present_sdms"
FUTURE_SDMS_DIR = "future_sdms"
SPECIES_TRAIT_FILE = "species_tbl.csv"
DISTURBANCE_FILE = "env_layers/area_swept_disturbance.tif"
FUTURE_DISTURBANCE_FILE = "env_layers/future_area_swept_disturbance.tif"
COST_FILE = "env_layers/cost.tif"
FUTURE_COST_FILE = "env_layers/future_cost.tif"
DATA_MASK = "env_layers/area_mask.npy"
results_dir = "results"

# Trained model and policy settings
TRAINED_MODEL = ""
N_TIME_STEPS = 50
TARGET_PROTECTED_CELLS = 17000
CELLS_PER_STEP = 1000

# Species parameters
AVG_CARRYING_CAPACITY = 100
DISPERSAL_RATE = 0.5  # can be an array (per-species values)
DISPERSAL_WINDOW = 3
MIN_HABITAT_SUITABILITY = 0.05  # can be an array (per-species values)

# Output
# Check data directory
if not DATA_DIR.exists() or str(DATA_DIR) == "/path/to/your/data":
    print("\nERROR: Please update DATA_DIR in this script to point to your data.")
    print("       See the example data repository for the expected format.")
    raise FileNotFoundError

os.makedirs(DATA_DIR / results_dir, exist_ok=True)

RES_DIR = DATA_DIR / results_dir
LOG_FILE = "training_log.tsv"
PLOT_DATA = True

# =============================================================================
# Episode Setup Function
# =============================================================================

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

# Load disturbance layer with predicted future change
disturbance = cn.load_spatial_data(
    file=DATA_DIR / DISTURBANCE_FILE,
    future_file=DATA_DIR / FUTURE_DISTURBANCE_FILE,
    mask=mask,
    lower_bound=0,
    upper_bound=1,
    n_time_steps=N_TIME_STEPS,
)

# Protection matrix (starts empty)
protection = cn.SpatialData(
    data=np.zeros(disturbance.shape),
    mask=mask,
    lower_bound=0,
    upper_bound=1,
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

# Load species traits
traits = cn.data_loader.load_trait_table(
    DATA_DIR / SPECIES_TRAIT_FILE,
    species_list=sdm.names,
    ref_column="species",
    fill_gaps=True,
)
# extract parameters for simulation
sensitivity = traits["sensitivity_disturbance"].to_numpy(copy=True)[:, np.newaxis]
growth_rates = traits["growth_rate"].to_numpy(copy=True) + 1.0
carrying_capacity = AVG_CARRYING_CAPACITY / traits["conservation_status"].to_numpy(
    copy=True
)
conservation_status = traits["conservation_status"].to_numpy(copy=True) - 1

# Initial extinction risk from conservation status
ext_risk = cn.ExtinctionRisk(
    init_status=conservation_status,
    n_classes=5,
    alpha=0.5,
)

# Load or create dispersal matrix (cached for efficiency)
disp_file = DATA_DIR / f"dispersal_d{DISPERSAL_RATE}_t{DISPERSAL_WINDOW}.npz"
if not disp_file.exists():
    print(f"Creating dispersal matrix: {disp_file}")
    cn.grid_utils.save_dispersal_distances(
        lambda_0=DISPERSAL_RATE,
        coords=sdm._coords,
        threshold=DISPERSAL_WINDOW,
        filename=str(disp_file),
    )
dispersal_matrix = cn.grid_utils.load_dispersal_distances(str(disp_file))

# Create environment
env = cn.BioEnv(
    sdms=sdm,
    disturbance=disturbance,
    costs=costs,
    protection_matrix=protection,
    species_k=carrying_capacity,
    growth_rates=growth_rates,
    sensitivity_rates=sensitivity,
    cached_dispersal_matrix=dispersal_matrix,
    ext_risk=ext_risk,
)

# Create agent components
feature_extractor = cn.FeatureExtractor(
    env,
    feature_set=None,  # Use default feature set (can be customized)
    time_rescale=N_TIME_STEPS / 2,
)

if PLOT_DATA:
    feature_extractor.plot_features(env, rescale=False, outdir=RES_DIR)

env.ext_risk.species_per_class(env.current_ext_risk)

model = cn.CellNN(input_dim=feature_extractor.n_features, hidden_dim=16)
policy = cn.PolicyNetwork(model, seed=SEED)
policy.set_flat_weights(np.load(TRAINED_MODEL))

rewards = cn.agents.rewards.NoRewards()

# Create episode runner
ep = cn.EpisodeRunner(
    env=env,
    feature_extractor=feature_extractor,
    policy_network=policy,
    rewards=rewards,
    n_steps=N_TIME_STEPS,
    n_total_protected_cells=TARGET_PROTECTED_CELLS,
    n_protected_cells_per_time_step=CELLS_PER_STEP,
    save_protection_history=True,
)

res, _ = ep.run_episode(np.load(TRAINED_MODEL))

cn.plots.plot_grid(
    # res["protection_matrix"][0]
    env.protection_matrix.reconstruct_grid[0],
    title="protection matrix",
    outfile=RES_DIR / "protection_matrix",
    dpi=300,
    figsize=(6, 8),
)

protection._data = protection.data + res["protection_history"]

cn.plots.plot_grid(
    protection.reconstruct_grid[0] + (2024 * (protection.reconstruct_grid[0] > 0)),
    title="protection matrix through time",
    outfile=RES_DIR / "protection_matrix_through_time",
    dpi=300,
    figsize=(6, 8),
    cmap="viridis",
)

# plot present extinction risks
cn.plots.plot_extinction_risk(
    env.ext_risk.init_status,
    labels=["LC", "NT", "VU", "EN", "CR"],
    outfile=RES_DIR / "Extinction_risk",
    title="Present extinction risk",
    dpi=200,
)

# plot (predicted) future extinction risks
cn.plots.plot_extinction_risk(
    env.current_ext_risk,
    labels=["LC", "NT", "VU", "EN", "CR"],
    outfile=RES_DIR / "Extinction_risk_future",
    title="Future extinction risk (protection)",
    dpi=200,
)

# run without protection for comparison
ep = cn.EpisodeRunner(
    env=env,
    feature_extractor=feature_extractor,
    policy_network=policy,
    rewards=rewards,
    n_steps=N_TIME_STEPS,
    n_total_protected_cells=0,
    n_protected_cells_per_time_step=0,
    save_protection_history=True,
)

res, _ = ep.run_episode(np.load(TRAINED_MODEL))

cn.plots.plot_extinction_risk(
    env.current_ext_risk,
    labels=["LC", "NT", "VU", "EN", "CR"],
    outfile=RES_DIR / "Extinction_risk_future_no_protection",
    title="Future extinction risk (no protection)",
    dpi=200,
)
