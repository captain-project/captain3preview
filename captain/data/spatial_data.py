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
from pyperlin import FractalPerlin2D

from captain.utils import data_loader, grid_utils

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
            nan_to_num: bool = True,  # force any remaining NaN to 0
            multi_per_step: np.ndarray | torch.Tensor | None = None,
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
            multi_per_step: Mulitplier of data values per time steps
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

        # Multiplier per step
        if multi_per_step is not None:
            if isinstance(multi_per_step, np.ndarray):
                multi_per_step = torch.from_numpy(multi_per_step.astype(np.float32))
            if multi_per_step.ndim == 2:
                multi_per_step = multi_per_step.unsqueeze()
            multi_np = multi_per_step.numpy()
            flat_multi, _, _ = grid_utils.flatten_grid(multi_np, mask)
            self._multi = torch.from_numpy(flat_multi).to(self.device, self.dtype)
        else:
            self._multi = None

        # Min threshold per channel
        self.reset_threshold(min_threshold)

        # Set NaNs to 0
        if nan_to_num:
            self._data.nan_to_num_(nan=0)
            self._delta.nan_to_num_(nan=0) if self._delta is not None else self._delta

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

    def reset_threshold(self, min_threshold: float) -> None:
        if min_threshold is not None:
            if isinstance(min_threshold, float):
                min_threshold = np.repeat(min_threshold, self._data_shape[0])
            if isinstance(min_threshold, np.ndarray):
                min_threshold = torch.from_numpy(min_threshold.astype(np.float32))
            self._min_threshold_per_channel = min_threshold.to(self.device, self.dtype)
        else:
            self._min_threshold_per_channel = None

    def update(self, time_step: int = 1) -> None:
        """Apply temporal evolution.

        Args:
            time_step: Current timestep multiplier for delta.
        """
        if self._delta is not None:
            self._data.add_(self._delta, alpha=time_step)
            self._data.clamp_(self._lower_bound, self._upper_bound)

        if self._multi is not None:
            self._data.mul_(self._multi)
            self._data.clamp_(self._lower_bound, self._upper_bound)

        self._step += 1

    def update_nonzero_mask(self) -> None:
        """Update the mask of cells with any non-zero values."""
        self._nonzero_cells_mask = torch.any(self._data > 0, dim=0)

    def update_cell_values(
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
    def reconstruct_grid_threshold(self) -> np.ndarray:
        """Create a 3D array for visualization (returns NumPy for plotting)."""
        return grid_utils.reconstruct_grid(
            self.data_min_threshold.cpu().numpy(), self._coords, self._data_shape[1:]
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


class StochasticSpatialData(SpatialData):
    def __init__(
            self,
            *args,
            risk_map: np.ndarray | torch.Tensor,
            noise_generator: FractalPerlin2D,
            binary_mask_2d: np.ndarray | torch.Tensor,
            **kwargs,
    ):
        """
        Subclass of SpatialData that handles risk-weighted, spatially coherent stochastic events.

        Args:
            risk_map: 2D or flattened tensor representing risk.
            noise_generator: Instantiated FractalPerlin2D generator.
            binary_mask_2d: 2D boolean tensor (True = valid simulation cell, False = NaN/boundary).
        """
        super().__init__(*args, **kwargs)

        # Convert numpy to torch if needed
        if isinstance(risk_map, np.ndarray):
            risk_map = torch.from_numpy(risk_map.astype(np.float32))
        self.risk_map = risk_map.to(self.device)
        self.noise_generator = noise_generator

        # Store the 2D mask explicitly so we can slice our 2D noise layers down to 1D
        if isinstance(binary_mask_2d, np.ndarray):
            binary_mask_2d = torch.from_numpy(binary_mask_2d)
        self.binary_mask_2d = binary_mask_2d.to(self.device, dtype=torch.bool)

    def apply_stochastic_events(self, intensity: float, impact_factor: float = 0.5):
        """Generates a coherent risk mask and applies an environmental degradation event."""
        # 1. Generate 2D Perlin noise: shape (1, H, W)
        noise_2d = self.noise_generator()[
            :, : self.binary_mask_2d.shape[0], : self.binary_mask_2d.shape[1]
        ]

        # Squeeze out the batch/channel dimension to get a clean (H, W) map
        noise_2d = noise_2d.squeeze(0)

        # 2. Threshold against risk map in 2D space
        # High risk areas require very little noise to trigger an event
        event_mask_2d = noise_2d < (1.0 - self.risk_map) * intensity

        # 3. CRITICAL BRIDGING STEP:
        # Filter the 2D event mask using our valid cell mask.
        # This collapses the (H, W) grid into a 1D vector of shape (valid_cells,)
        # matching your internal self._data structure perfectly.
        flat_event_mask = event_mask_2d[self.binary_mask_2d]

        # 4. Apply the event in-place to all channels of our valid data
        # For example, scaling habitat suitability down by our impact factor where events hit
        self._data[:, flat_event_mask] *= impact_factor

    def update(
            self, time_step: int = 1, trigger_events: bool = True, intensity: float = 0.3
    ):
        # reset previous steps
        self.reset()

        # If requested for this simulation step, layer the stochastic event on top
        if trigger_events:
            self.apply_stochastic_events(intensity=intensity)


def plot_data_evolution(
        data: SpatialData | StochasticSpatialData,
        n_steps: int = 20,
        indx: int = 0,
        skip: int = 1,
        title: str | None = None,
        outfile: str | Path | None = None,
        cmap: str = "YlGnBu",
        vmin: float | None = None,
        vmax: float | None = None,
        apply_min_threshold: bool = True,
        create_gif: bool = True,
        remove_png: bool = True,
        frame_duration_ms: int = 100,
        gif_duration_sec: int | None = None,
        figsize=(5, 6),
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
        f_name = (
            str(outfile) + "_t" + str(i).zfill(3) + ".png"
            if outfile is not None
            else None
        )
        dat.update()
        if i % skip == 0:

            if apply_min_threshold:
                d = dat.reconstruct_grid_threshold[indx]
            else:
                d = dat.reconstruct_grid[indx]

            plots.plot_grid(
                d,
                title=f"{title} - time step {i}",
                outfile=f_name,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                figsize=figsize,
            )
            file_names.append(f_name)
    if create_gif:
        if gif_duration_sec is not None:
            frame_duration_ms = gif_duration_sec * 1000 / len(file_names)
        plots.create_gif(file_names, duration_ms=frame_duration_ms, rm_png=remove_png)


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
        nan_to_num: bool = True,  # force any remaining NaN to 0
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

    else:
        delta_sdm = None

    dat = SpatialData(
        data=maps,
        mask=mask,
        delta_per_step=delta_sdm,
        names=names,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        min_threshold=min_threshold,
        nan_to_num=nan_to_num,
    )

    return dat
