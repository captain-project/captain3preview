# Inference Outputs for xAI

When running a trained policy, `EpisodeRunner` can save the features the policy saw, the
scores it produced and the resulting cell ranking at each decision. These files are meant for
explainability analyses (e.g. feature attribution, partial dependence, surrogate models).

Recording is off by default and has no cost during training.

## Usage

```python
import captain as cn

recorder = cn.InferenceRecorder(
    "results/xai",
    steps="all",             # or a list of time steps, e.g. [0] for the initial ranking
    save_raw_features=True,  # also save features before normalization
    compress=False,          # np.savez_compressed if True
    feature_sample_fraction=None,  # e.g. 0.05 to save features for a subset of cells
    seed=None,               # seed for the random cell sample
)

runner = cn.EpisodeRunner(
    env=env,
    feature_extractor=feature_extractor,
    policy_network=policy,
    rewards=cn.NoRewards(),
    budget_manager=budget_manager,
    n_steps=N_TIME_STEPS,
    recorder=recorder,
)
runner.run_episode(weights)
```

`BridgeEpisodeRunner` accepts the same `recorder` argument. In `examples/run_inference.py`
this is controlled by `SAVE_XAI_DATA` and `XAI_STEPS`.

## Output files

One file per policy decision, `step_{t:03d}_u{u:02d}.npz`, where `t` is the time step and `u`
the feature update within that time step (always `00` unless
`feature_updates_per_time_step > 1`). `F` = number of features, `N` = number of cells.

| Key | Shape | Description |
|-----|-------|-------------|
| `features_input` | (F, N) or (F, M) | Features as fed to the network (z-scored, clipped) |
| `features_raw` | (F, N) or (F, M) | Features before normalization (`FeatureExtractor` only) |
| `feature_names` | (F,) | Feature names |
| `sample_cell_idx` | (M,) | Cells whose features are saved (subsampling only) |
| `sample_is_top` | (M,) | True for top-ranked cells, False for random cells (subsampling only) |
| `sample_inclusion_prob` | (M,) | Probability that the cell was sampled (subsampling only) |
| `scores` | (N,) | Raw network output, before masking |
| `eligible` | (N,) | Cells that could be selected at this decision |
| `already_protected` | (N,) | Cells protected before this decision |
| `selected` | (N,) | Cells selected at this decision |
| `rank` | (N,) | 1 = highest priority among eligible cells, -1 = not eligible |
| `region_id` | (N,) | Index into `region_names`, -1 = no region (regional agents only) |
| `time_step`, `update_idx` | scalar | Decision indices |

With regional agents, `rank` is computed within each region, matching how cells are selected.
The selected cells are the top-`k` ranked cells (ignoring the negligible tie-break noise).

## Feature subsampling

Features dominate file size (`2 × F × N` float32 values per decision). With
`feature_sample_fraction` set, `scores`, `rank` and the masks are still saved for all cells,
but features are saved only for a sample of cells, drawn at each decision among eligible cells
(already protected and masked cells are excluded):

- **Top cells:** the `n` best-ranked eligible cells, with
  `n = max(ceil(fraction × n_eligible), n_selected)`, so the selected cells are always included.
- **Random cells:** `n` cells drawn at random from the remaining eligible cells (all of them if
  fewer than `n` remain).

With regional agents this is done within each region. Column `j` of the feature arrays belongs
to cell `sample_cell_idx[j]`. Top cells have `sample_inclusion_prob = 1`, random cells
`n / n_remaining`; the sample over-represents top cells, so weight rows by
`1 / sample_inclusion_prob` when estimating statistics over all eligible cells.

```python
step = np.load("results/xai/step_000_u00.npz")
X = step["features_input"].T                        # (M, F)
y = step["sample_is_top"]                           # top vs rest
w = 1.0 / step["sample_inclusion_prob"]             # reweighting to all eligible cells
```

A single `cells.npz` stores `coords_row`, `coords_col`, `grid_shape`, `feature_names` and, for
regional agents, `region_names`.

> **Note:** once the protection budget is exhausted the policy is no longer queried, so later
> time steps produce no files. A warning lists any requested steps without a decision.

## Mapping back to the grid

```python
import numpy as np
from captain.utils import grid_utils

cells = np.load("results/xai/cells.npz")
step = np.load("results/xai/step_000_u00.npz")

coords = (cells["coords_row"], cells["coords_col"])
rank_map = grid_utils.reconstruct_grid(step["rank"][None], coords, tuple(cells["grid_shape"]))[0]
```

When using `CellNN`, scores are computed independently per cell from its feature vector, so
`features_input` (transposed to `(N, F)`) and `scores` form a complete input/output dataset for
the policy.
