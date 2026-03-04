import glob
import os
import warnings

import numpy as np
import pandas as pd
import rasterio
import rioxarray as rxr


def load_map(
    filename,
    clip_min: float | None = None,
    clip_max: float | None = None,
    nan_to_num: bool = False,
):
    # print("Getting SDM data", sdm_files[taxon_index])
    extension = os.path.splitext(filename)[1]

    if extension == ".tif":
        warnings.filterwarnings(
            "ignore", category=rasterio.errors.NotGeoreferencedWarning
        )
        warnings.filterwarnings(
            "ignore",
            message="angle from rectified to skew grid parameter lost in conversion to CF",
        )

        data = rxr.open_rasterio(filename).to_numpy()[0]
        name = os.path.basename(filename).split(".tif")[0]
    elif extension == ".npy":
        data = np.load(filename)
        name = os.path.basename(filename).split(".npy")[0]
    else:
        raise ValueError(f"Unrecognized extension: {extension}, {filename}")

    data = np.clip(data, clip_min, clip_max)
    if nan_to_num:
        np.nan_to_num(data, copy=False)

    return data, name


def load_maps_from_dir(
    dir,
    extension: str = "",
    clip_min: float | None = None,
    clip_max: float | None = None,
):
    filenames = np.sort(glob.glob(os.path.join(dir, "*" + extension)))
    if len(filenames) == 0:
        raise ValueError(f"\nNo files found, check path: \n{dir}\n")
    data = []
    names = []
    for filename in filenames:
        dat, name = load_map(filename)
        data.append(dat)
        names.append(name)

    data = np.squeeze(np.array(data))
    data = np.clip(data, clip_min, clip_max)

    return data, np.array(names)


def load_trait_table(
    filename,
    species_list: list | np.ndarray,
    ref_column: str | None = None,
    fill_gaps: bool = False,
):
    trait_table = pd.read_csv(filename)
    if ref_column is None:
        ref_column = trait_table.columns[0]
    tbl = reorder_by_species(trait_table, species_list, ref_column)

    if fill_gaps:
        # 1. Calculate means for all numeric columns
        means = tbl.mean(numeric_only=True)
        # 2. Fill the whole dataframe using the means dictionary
        tbl.fillna(means, inplace=True)
        for col in tbl.select_dtypes(include="number").columns:
            # This is safe because it explicitly re-assigns the data
            tbl[col] = tbl[col].fillna(tbl[col].mean())

    return tbl


def reorder_by_species(df, species_list, ref_column: str):
    """
    Reorders a DataFrame based on a specific list of species names.
    Checks for mismatches in both directions.
    """
    # Convert species_name to a set for faster lookup
    df_species = set(df[ref_column])
    target_species = set(species_list)

    # 1. Check for mismatches
    # Species in your target list but missing from the DataFrame
    missing_in_df = target_species - df_species
    # Species in the DataFrame but not in your target list
    missing_in_list = df_species - target_species

    if missing_in_df or missing_in_list:
        if missing_in_df:
            print(
                f"Error: The following species from the target list are missing in the DataFrame: {missing_in_df}"
            )
        if missing_in_list:
            print(
                f"Error: The following species in the DataFrame are not in the target list: {missing_in_list}"
            )
        raise ValueError(
            f"Mismatches found between species_list and DataFrame {ref_column} column."
        )

    # 2. Reorder the rows
    # We set species_name as the index, reindex it, and then reset it back to a column
    df_reordered = df.set_index(ref_column).reindex(species_list).reset_index()

    return df_reordered
