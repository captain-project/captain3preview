#!/usr/bin/env python
"""Example: Infer protection matrix from trained model.

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

import logging
import warnings

# Filter out the specific PyTorch Sparse CSR beta warning
warnings.filterwarnings("ignore", message="Sparse CSR tensor support is in beta state")
import os
from pathlib import Path

import numpy as np
import pandas as pd

import captain as cn

# Configure logging to print INFO messages to your console/Slurm log
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()],  # Sends output to the terminal/stderr
)

SEED = None
# =============================================================================
# Configuration
# =============================================================================

# Data paths - the repo's own toy dataset, resolved relative to this script
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

PRESENT_SDMS_DIR = "present_sdms"
FUTURE_SDMS_DIR = "future_sdms"
SPECIES_TRAIT_FILE = "species_tbl.csv"
DISTURBANCE_FILE = "env_layers/area_swept_disturbance.tif"
FUTURE_DISTURBANCE_FILE = "env_layers/future_area_swept_disturbance.tif"
COST_FILE = "env_layers/cost.tif"
FUTURE_COST_FILE = "env_layers/future_cost.tif"
DATA_MASK = "env_layers/area_mask.npy"

# Regional-agent setup - must match the config used to train the model being
# loaded (see train_policy.py): USE_REGIONAL_AGENTS=True loads a coordinated
# per-region policy. When False, REGION_ID selects a single global agent
# (None = unrestricted, or a region NAME to restrict that single agent to
# just that region).
USE_REGIONAL_AGENTS = False
REGION_ID = None
REGION_FILE = "env_layers/regions.tif"
REGION_TABLE = "env_layers/regions_tbl.csv"
REWARD_WEIGHTS = np.array([1.0, 1.0])

# Trained model and policy settings
N_TIME_STEPS = 50
TARGET_PROTECTED_CELLS_FRACTION = 0.10  # Fraction of valid cells to be protected
CELLS_PER_STEP = 1000

# Species parameters
AVG_CARRYING_CAPACITY = 100
DISPERSAL_RATE = 0.5  # can be an array (per-species values)
DISPERSAL_WINDOW = 3
MIN_HABITAT_SUITABILITY = None  # Overwritten below by species-specific thresholds

# Output - locate the trained model using the same directory-naming formula
# train_policy.py uses for RESULTS_DIR, and write inference outputs into a
# subfolder of it (so training/inference outputs for a given config stay
# together and can never drift out of sync).
if USE_REGIONAL_AGENTS:
    res_name = "regions"
elif REGION_ID is None:
    res_name = "global"
else:
    res_name = REGION_ID

MODEL_DIR = DATA_DIR / (
        res_name
        + "_w"
        + "_w".join([str(r) for r in REWARD_WEIGHTS])
        + f"_p{TARGET_PROTECTED_CELLS_FRACTION}"
)
TRAINED_MODEL = MODEL_DIR / "trained_weights.npy"
RES_DIR = MODEL_DIR / "inference"
os.makedirs(RES_DIR, exist_ok=True)

LOG_FILE = "training_log.tsv"
PLOT_DATA = True

# xAI output: save features, policy scores and cell rankings at each decision
# (one npz per decision in RES_DIR / "xai", see cn.InferenceRecorder)
SAVE_XAI_DATA = True
XAI_STEPS = "all"  # "all", or a list of time steps, e.g. [0] for the initial ranking
# None = save features for all cells; e.g. 0.05 = save features only for the top
# max(5%, selected) eligible cells plus an equally sized random sample of the rest
XAI_SAMPLE_FRACTION = 0.05

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

# Regional-agent setup: build per-region masks/targets, or a single
# action_mask restricting a global agent to one region.
TARGET_PROTECTED_CELLS = int(TARGET_PROTECTED_CELLS_FRACTION * np.nansum(mask))

if USE_REGIONAL_AGENTS:
    tmp, _ = cn.data_loader.load_map(DATA_DIR / REGION_FILE)
    region_tbl = pd.read_csv(DATA_DIR / REGION_TABLE)

    regional_totals, per_step_regional_targets, region_masks = {}, {}, {}
    for region_num, name in zip(region_tbl["REGION_ID"], region_tbl["NAME"]):
        r_mask = cn.SpatialData(
            data=tmp == region_num, mask=mask, lower_bound=0, upper_bound=1
        )
        target = int(TARGET_PROTECTED_CELLS_FRACTION * r_mask.data.sum())
        regional_totals[name] = target
        per_step_regional_targets[name] = int(
            CELLS_PER_STEP * (r_mask.data.sum() / np.nansum(mask))
        )
        region_masks[name] = r_mask._nonzero_cells_mask
        print(
            f"Region {region_num} ({name}): size={r_mask.data.sum()}, "
            f"target={target}, per_step={per_step_regional_targets[name]}"
        )

    region_mask = None  # regional mode does not restrict env.action_mask

elif REGION_ID is None:
    region_mask = None

else:
    tmp, _ = cn.data_loader.load_map(DATA_DIR / REGION_FILE)
    region_tbl = pd.read_csv(DATA_DIR / REGION_TABLE)
    match = region_tbl.loc[region_tbl["NAME"] == REGION_ID, "REGION_ID"]
    if match.empty:
        raise ValueError(f"REGION_ID {REGION_ID!r} not found in {REGION_TABLE}")
    region_num = match.iloc[0]

    region_mask = cn.SpatialData(
        data=tmp != region_num, mask=mask, lower_bound=0, upper_bound=1
    )
    TARGET_PROTECTED_CELLS = int(
        TARGET_PROTECTED_CELLS_FRACTION * (1 - region_mask.data).sum()
        + protection.data.sum()
    )

# Load species traits
traits = cn.data_loader.load_trait_table(
    DATA_DIR / SPECIES_TRAIT_FILE,
    species_list=sdm.names,
    ref_column="species",
    fill_gaps=True,
)

# Per-species minimum habitat suitability (must match training)
sdm.reset_threshold(traits["min_habitat_suitability"].to_numpy())

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
    action_mask=region_mask,  # None, or restricts actions to one region
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
if USE_REGIONAL_AGENTS:
    policy = cn.RegionalPolicyNetwork(model, seed=SEED)
else:
    policy = cn.PolicyNetwork(model, seed=SEED)
policy.set_flat_weights(np.load(TRAINED_MODEL))

rewards = cn.NoRewards()

# Create episode runner
if USE_REGIONAL_AGENTS:
    budget_manager = cn.RegionalBudgetManager(
        masks=region_masks,
        total_targets=regional_totals,
        cells_per_time_step=per_step_regional_targets,
    )
else:
    budget_manager = cn.GlobalBudgetManager(
        total_target=TARGET_PROTECTED_CELLS,
        cells_per_time_step=CELLS_PER_STEP,
        feature_updates_per_time_step=1,
    )

ep = cn.EpisodeRunner(
    env=env,
    feature_extractor=feature_extractor,
    policy_network=policy,
    rewards=rewards,
    n_steps=N_TIME_STEPS,
    budget_manager=budget_manager,
    save_protection_history=True,
    recorder=cn.InferenceRecorder(
        RES_DIR / "xai", steps=XAI_STEPS, feature_sample_fraction=XAI_SAMPLE_FRACTION, seed=SEED
    ) if SAVE_XAI_DATA else None,
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

history = (res["protection_history"] > 0).int() * (
        1 + res["protection_history"].max() - res["protection_history"]
)
protection_res = cn.SpatialData(
    data=np.zeros(disturbance.shape),
    mask=mask,
    lower_bound=0,
    upper_bound=1,
)
protection_res._data += history

cn.plots.plot_grid(
    protection_res.reconstruct_grid[0] + (2024 * (protection.reconstruct_grid[0] > 0)),
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
# (NoBudgetManager's step context isn't compatible with RegionalPolicyNetwork,
# which requires region_masks/region_k rather than n_cells, so skip this
# comparison run in "All"-regions mode.)
if not USE_REGIONAL_AGENTS:
    ep = cn.EpisodeRunner(
        env=env,
        feature_extractor=feature_extractor,
        policy_network=policy,
        rewards=rewards,
        n_steps=N_TIME_STEPS,
        budget_manager=cn.NoBudgetManager(),
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
else:
    print(
        "Skipping no-protection comparison run in regional multi-agent mode "
        "(RegionalPolicyNetwork requires region_masks/region_k, not n_cells)."
    )
