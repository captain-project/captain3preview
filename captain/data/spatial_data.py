"""Spatial data container for gridded environmental data.

This module provides the SpatialData class for efficient storage and manipulation
of spatial data (e.g., habitat suitability, disturbance, costs) used in the
biodiversity simulation.
"""

from __future__ import annotations

import copy
import logging
import warnings
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

from captain.utils import grid_utils, data_loader

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class SpatialData:
    """Container for spatial gridded data with temporal evolution support.

    Stores spatial data in a memory-efficient flattened format, keeping only
    valid (non-masked) cells. Supports temporal evolution via delta per step,
    with optional memory-mapped backup for large datasets.

    Attributes:
        data: Tensor of shape (channels, valid_cells) containing the data.
        shape: Original 3D shape (channels, x, y) before flattening.
        names: Names for each channel.

    Example:
        >>> data = torch.rand(10, 100, 100)  # 10 species, 100x100 grid
        >>> spatial = SpatialData(data, device="cuda")
        >>> spatial.update(timestep=1)  # Apply temporal evolution
        >>> spatial.reset()  # Return to initial state
    """

    def __init__(
            self,
            data: np.ndarray | torch.Tensor,
            mask: np.ndarray | None = None,
            delta_per_step: np.ndarray | torch.Tensor | None = None,
            lower_bound: float | None = None,
            upper_bound: float | None = None,
            backup_path: str | Path | None = None,
            min_threshold: float | np.ndarray | torch.Tensor | None = None,
            names: list[str] | np.ndarray | None = None,
            device: torch.device | str = "cpu",
            dtype: torch.dtype = torch.float32,
    ):
        """Initialize SpatialData container.

        Args:
            data: Input data of shape (channels, x, y) or (x, y).
            mask: Optional mask array; NaN values indicate invalid cells.
            delta_per_step: Rate of change per timestep, same shape as data.
            lower_bound: Minimum value after update (default: -inf).
            upper_bound: Maximum value after update (default: inf).
            backup_path: Path to save initial state for memory-mapped reset.
            min_threshold: Per-channel minimum threshold for data_min_threshold.
            names: Names for each channel.
            device: PyTorch device ('cpu', 'cuda', 'mps').
            dtype: PyTorch dtype (default: float32).
        """
        self.device = torch.device(device)
        self.dtype = dtype

        # Convert numpy to torch if needed
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data.astype(np.float32))

        # Add channel axis if missing
        if data.ndim == 2:
            data = data.unsqueeze(0)

        # Store original shape for reconstruction
        self._data_shape = tuple(data.shape)

        # Flatten grid to only valid cells
        data_np = data.numpy()
        flat_data, self._coords, _ = grid_utils.flatten_grid(data_np, mask)
        self._data = torch.from_numpy(flat_data).to(self.device, self.dtype)

        # Bounds
        self._lower_bound = lower_bound if lower_bound is not None else float("-inf")
        self._upper_bound = upper_bound if upper_bound is not None else float("inf")

        # Delta per step
        if delta_per_step is not None:
            if isinstance(delta_per_step, np.ndarray):
                delta_per_step = torch.from_numpy(delta_per_step.astype(np.float32))
            if delta_per_step.ndim == 2:
                delta_per_step = delta_per_step.unsqueeze(0)
            delta_np = delta_per_step.numpy()
            flat_delta, _, _ = grid_utils.flatten_grid(delta_np, mask)
            self._delta = torch.from_numpy(flat_delta).to(self.device, self.dtype)
        else:
            self._delta = None

        # Min threshold per channel
        if min_threshold is not None:
            if isinstance(min_threshold, float):
                min_threshold = np.repeat(min_threshold, self._data_shape[0])
            if isinstance(min_threshold, np.ndarray):
                min_threshold = torch.from_numpy(min_threshold.astype(np.float32))
            self._min_threshold_per_channel = min_threshold.to(self.device, self.dtype)
        else:
            self._min_threshold_per_channel = None

        self._step = 0

        # Backup handling
        self._backup_path = Path(backup_path) if backup_path is not None else None
        if self._backup_path is not None:
            np.save(self._backup_path, self._data.cpu().numpy())
            self._initial_data = None
        else:
            self._initial_data = self._data.clone()

        # Channel names
        if names is not None:
            self._names = (
                [str(i) for i in names] if not isinstance(names, list) else names
            )
        else:
            self._names = [f"dat_{i}" for i in range(self._data_shape[0])]

        self.update_nonzero_mask()

    def reset(self) -> None:
        """Reset data to initial state."""
        if self._initial_data is not None:
            self._data.copy_(self._initial_data)
        else:
            # Reload from stored file
            initial = np.load(self._backup_path, mmap_mode="r")
            self._data.copy_(
                torch.from_numpy(initial.copy()).to(self.device, self.dtype)
            )
        self._step = 0
        self.update_nonzero_mask()

    def update(self, time_step: int = 1) -> None:
        """Apply temporal evolution.

        Args:
            time_step: Current timestep multiplier for delta.
        """
        if self._delta is not None:
            self._data.add_(self._delta, alpha=time_step)
            self._data.clamp_(self._lower_bound, self._upper_bound)
            self._step += 1

    def update_nonzero_mask(self) -> None:
        """Update the mask of cells with any non-zero values."""
        self._nonzero_cells_mask = torch.any(self._data > 0, dim=0)

    def update_col_values(
            self, idx: torch.Tensor | np.ndarray | list, val: float
    ) -> None:
        """Set values for specific columns (cells).

        Args:
            idx: Indices of columns to update.
            val: Value to set.
        """
        if isinstance(idx, np.ndarray):
            idx = torch.from_numpy(idx)
        elif isinstance(idx, list):
            idx = torch.tensor(idx)
        self._data[:, idx] = val
        self.update_nonzero_mask()

    @property
    def reconstruct_grid(self) -> np.ndarray:
        """Create a 3D array for visualization (returns NumPy for plotting)."""
        return grid_utils.reconstruct_grid(
            self._data.cpu().numpy(), self._coords, self._data_shape[1:]
        )

    @property
    def data(self) -> torch.Tensor:
        """Current data tensor of shape (channels, valid_cells)."""
        return self._data

    @property
    def shape(self) -> tuple[int, ...]:
        """Original 3D shape (channels, x, y)."""
        return self._data_shape

    @property
    def data_min_threshold(self) -> torch.Tensor:
        """Data with per-channel minimum threshold applied."""
        if self._min_threshold_per_channel is not None:
            threshold_mask = self._data >= self._min_threshold_per_channel.unsqueeze(1)
            return self._data * threshold_mask.to(self.dtype)
        return self._data

    @property
    def names(self) -> list[str]:
        """Channel names."""
        return self._names

    def to(self, device: torch.device | str) -> SpatialData:
        """Move data to specified device.

        Args:
            device: Target device.

        Returns:
            Self for chaining.
        """
        self.device = torch.device(device)
        self._data = self._data.to(self.device)
        if self._delta is not None:
            self._delta = self._delta.to(self.device)
        if self._initial_data is not None:
            self._initial_data = self._initial_data.to(self.device)
        if self._min_threshold_per_channel is not None:
            self._min_threshold_per_channel = self._min_threshold_per_channel.to(
                self.device
            )
        self._nonzero_cells_mask = self._nonzero_cells_mask.to(self.device)
        return self


def plot_data_evolution(
        data: SpatialData,
        n_steps: int = 20,
        indx: int = 0,
        skip: int = 1,
        title: str | None = None,
        outfile: str | Path | None = None,
        cmap: str = "YlGnBu",
        vmin: float | None = None,
        vmax: float | None = None,
        create_gif: bool = True,
        remove_png: bool = True,
        duration_ms=100,
) -> None:
    """Plot temporal evolution of spatial data.

    Args:
        data: SpatialData instance to plot.
        n_steps: Number of steps to simulate.
        indx: Channel index to plot.
        skip: Plot every nth step.
        title: Plot title (default: channel name).
    """
    from captain.utils import plots

    dat = copy.deepcopy(data)
    skip = max(skip, 1)
    dat.reset()
    file_names = []
    if title is None:
        title = dat.names[indx]
    for i in range(n_steps):
        if outfile is not None:
            f_name = str(outfile) + "_t" + str(i).zfill(3) + ".png"

        else:
            f_name = None
        dat.update()
        if i % skip == 0:
            plots.plot_grid(
                dat.reconstruct_grid[indx],
                title=f"{title} - time step {i}",
                outfile=f_name,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
            )
            file_names.append(f_name)
    if create_gif:
        plots.create_gif(file_names, duration_ms=duration_ms, rm_png=remove_png)


def load_spatial_data(
        file: str | Path,
        future_file: str | Path | None = None,
        mask: np.ndarray | None = None,
        lower_bound: float | None = None,
        upper_bound: float | None = None,
        n_time_steps: int | None = None,
        min_threshold: float | np.ndarray | torch.Tensor | None = None,
        nan_to_num: bool = True,
) -> SpatialData:
    maps, names = data_loader.load_map(
        file, clip_min=lower_bound, clip_max=upper_bound, nan_to_num=nan_to_num
    )
    if future_file is not None:
        maps_future, names_future = data_loader.load_map(
            future_file,
            clip_min=lower_bound,
            clip_max=upper_bound,
            nan_to_num=nan_to_num,
        )

        # Calculate per-step change in habitat suitability
        if n_time_steps is not None:
            delta_sdm = grid_utils.calculate_delta(maps, maps_future, n_time_steps)
        else:
            raise ValueError("n_time_steps not specified")
    else:
        delta_sdm = None

    dat = SpatialData(
        data=maps,
        mask=mask,
        delta_per_step=delta_sdm,
        names=names,
        lower_bound=0,
        upper_bound=1,
        min_threshold=min_threshold,
    )

    return dat


def load_spatial_data_from_dir(
        dir: str | Path,
        future_dir: str | Path | None = None,
        extension: str = "",
        mask: np.ndarray | None = None,
        lower_bound: float | None = None,
        upper_bound: float | None = None,
        n_time_steps: int | None = None,
        min_threshold: float | np.ndarray | torch.Tensor | None = None,
) -> SpatialData:
    maps, names = data_loader.load_maps_from_dir(
        dir,
        clip_min=lower_bound,
        clip_max=upper_bound,
        extension=extension,
    )
    if future_dir is not None:
        maps_future, names_future = data_loader.load_maps_from_dir(
            future_dir,
            clip_min=lower_bound,
            clip_max=upper_bound,
            extension=extension,
        )

        # Calculate per-step change in habitat suitability
        if n_time_steps is not None:
            delta_sdm = grid_utils.calculate_delta(maps, maps_future, n_time_steps)
        else:
            raise ValueError("n_time_steps not specified")

        if np.all(names_future == names):
            pass
        else:
            warnings.warn(
                "\nNames of loaded maps do not match, assuming they correspond."
            )
            warnings.filterwarnings(
                "ignore",
                message="\nNames of loaded maps do not match, assuming they correspond.",
            )
            # [
            #     print(i)
            #     for i in zip(
            #         names[: np.minimum(len(names), 5)],
            #         names_future[: np.minimum(len(names), 5)],
            #     )
            # ]

    else:
        delta_sdm = None

    dat = SpatialData(
        data=maps,
        mask=mask,
        delta_per_step=delta_sdm,
        names=names,
        lower_bound=0,
        upper_bound=1,
        min_threshold=min_threshold,
    )

    return dat
