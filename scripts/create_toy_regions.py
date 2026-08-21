#!/usr/bin/env python
"""Generate a synthetic region raster for the toy dataset.

The toy dataset (unlike the Swiss example) has no canton/region-equivalent
raster, but `train_policy.py` / `run_inference.py` need one to demonstrate
regional multi-agent training (`RegionalPolicyNetwork` + `RegionalBudgetManager`).

This script partitions the toy grid into a 3x3 grid of rectangular regions,
writing:
- data/env_layers/regions.tif   (region-ID raster, georeferenced like cost.tif)
- data/env_layers/regions_tbl.csv (REGION_ID -> NAME lookup table)

Run once as a data-prep step; the outputs are consumed by the other scripts.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

REFERENCE_TIF = DATA_DIR / "env_layers" / "cost.tif"
MASK_FILE = DATA_DIR / "env_layers" / "area_mask.npy"
OUTPUT_TIF = DATA_DIR / "env_layers" / "regions.tif"
OUTPUT_CSV = DATA_DIR / "env_layers" / "regions_tbl.csv"

N_ROWS = 3
N_COLS = 3
NODATA_VALUE = -1.0


def main():
    with rasterio.open(REFERENCE_TIF) as src:
        profile = src.profile.copy()
        height, width = src.height, src.width
        ref_crs, ref_transform = src.crs, src.transform

    mask = np.load(MASK_FILE)

    # Partition the bounding box of valid (mask==1) cells rather than the full
    # canvas: the toy mask only covers a small, irregular sub-area of the grid,
    # so tiling the full extent would leave some tiles with ~0 valid cells.
    valid_rows = np.where(mask.sum(axis=1) > 0)[0]
    valid_cols = np.where(mask.sum(axis=0) > 0)[0]
    row_groups = np.array_split(np.arange(valid_rows.min(), valid_rows.max() + 1), N_ROWS)
    col_groups = np.array_split(np.arange(valid_cols.min(), valid_cols.max() + 1), N_COLS)

    region_grid = np.full((height, width), NODATA_VALUE, dtype=np.float32)
    names = []
    ids = []
    for i, rows in enumerate(row_groups):
        for j, cols in enumerate(col_groups):
            region_id = i * N_COLS + j + 1
            region_grid[np.ix_(rows, cols)] = region_id
            ids.append(region_id)
            names.append(f"tile_r{i + 1}_c{j + 1}")

    # Respect the area mask (no-op today since the toy mask is all-valid, but
    # keeps this script correct if the mask is ever tightened).
    region_grid[mask == 0] = NODATA_VALUE

    profile.update(dtype="float32", count=1, nodata=NODATA_VALUE)
    with rasterio.open(OUTPUT_TIF, "w", **profile) as dst:
        dst.write(region_grid, 1)

    pd.DataFrame({"REGION_ID": ids, "NAME": names}).to_csv(OUTPUT_CSV, index=False)

    # Self-check: confirm the written raster matches the reference georeferencing.
    with rasterio.open(OUTPUT_TIF) as src:
        assert src.shape == (height, width), "shape mismatch"
        assert src.crs == ref_crs, "CRS mismatch"
        assert src.transform == ref_transform, "transform mismatch"

    print(f"Wrote {OUTPUT_TIF} and {OUTPUT_CSV} ({len(ids)} regions)")
    for region_id, name in zip(ids, names):
        count = int(np.sum(region_grid == region_id))
        print(f"  {name} (id={region_id}): {count} cells")


if __name__ == "__main__":
    main()
