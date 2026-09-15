"""CNN-based feature extraction from environment state.

Replaces hand-engineered features with a learned SpatialCNN that processes
(H, W) spatial grids. Outputs per-cell features with the same interface as
FeatureExtractor.observe().
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from captain.agents.spatial_cnn import SpatialCNN
    from captain.environment.bioenv import BioEnv

logger = logging.getLogger(__name__)

# Number of hand-engineered input channels packed into the (1, C, H, W) grid
# fed into the SpatialCNN. See FeatureExtractorCNN docstring for the channel
# layout.
N_INPUT_CHANNELS: int = 12


class FeatureExtractorCNN:
    """Extract per-cell features using a learned SpatialCNN.

    Builds a (1, 12, H, W) grid from BioEnv state each timestep, runs it
    through a SpatialCNN, and extracts valid-cell features.

    Input channels (12 total):
        0: Species richness per cell
        1: Total population (log-scaled)
        2-6: Extinction risk per class (one-hot x presence, 5 channels)
        7: Disturbance (mean across channels)
        8: Cost (mean across channels)
        9: Protection status (binary)
        10: Time (broadcast scalar)
        11: Valid cell mask (binary)

    Per-channel z-score normalization (computed once from initial state)
    is applied before the CNN.

    Attributes:
        n_features: Number of output features per cell (from CNN).
        device: PyTorch device.

    Example:
        >>> from captain.agents.spatial_cnn import SpatialCNN
        >>> cnn = SpatialCNN(12, 8, 8)
        >>> extractor = FeatureExtractorCNN(env, cnn=cnn)
        >>> features = extractor.observe(env)  # shape: (8, n_cells)
    """

    CLIP_RANGE: float = 5.0
    DEFAULT_TIME_RESCALE: float = 10.0

    def __init__(
        self,
        env: BioEnv,
        cnn: SpatialCNN,
        static_features: np.ndarray | torch.Tensor | None = None,
        time_rescale: float = DEFAULT_TIME_RESCALE,
        device: torch.device | str | None = None,
    ):
        """Initialize CNN feature extractor.

        Args:
            env: Environment to extract features from.
            cnn: SpatialCNN module (shared with CellCNNPolicy).
            time_rescale: Divisor for time normalization.
            device: PyTorch device (default: same as env).
        """
        self.device = torch.device(device) if device else env.device
        self.cnn = cnn.to(self.device)
        self._time_rescale = time_rescale
        self._n_risk_classes = env.ext_risk._n_classes

        # Grid dimensions from SpatialData
        self._grid_h, self._grid_w = env.sdms._data_shape[1:]
        self._n_cells = env.n_cells

        # Coordinate indices for mapping between flat cells and 2D grid
        coords = env.sdms._coords
        self._row_coords = torch.tensor(coords[0], dtype=torch.long, device=self.device)
        self._col_coords = torch.tensor(coords[1], dtype=torch.long, device=self.device)

        if isinstance(static_features, (np.ndarray, torch.Tensor)):
            self._static_features = torch.as_tensor(static_features).to(self.device)
        else:
            self._static_features = None

        # Pre-allocate grid buffer: (1, 12, H, W)
        self._grid = torch.zeros(
            (
                1,
                self.cnn.n_input_channels,
                self._grid_h,
                self._grid_w,
            ),
            device=self.device,
            dtype=torch.float32,
        )

        if self._static_features is not None:
            # add static features
            self._grid[
                0,
                -(1 + self._static_features.size(0)) : -1,
                self._row_coords,
                self._col_coords,
            ] = self._static_features

        # Valid cell mask (channel 11) is static — fill once
        self._grid[0, -1, self._row_coords, self._col_coords] = 1.0

        # Compute normalization stats from initial state
        self._set_rescaler(env)

    @property
    def n_features(self) -> int:
        """Number of output features per cell."""
        return self.cnn.n_output_features

    @property
    def feature_names(self) -> list[str]:
        """Generic names of the (learned) CNN output features."""
        return [f"cnn_{k}" for k in range(self.n_features)]

    def _fill_grid(self, env: BioEnv) -> torch.Tensor:
        """Build the (1, 12, H, W) input grid from environment state.

        Args:
            env: Current environment.

        Returns:
            Grid tensor of shape (1, 12, H, W).
        """
        g = self._grid
        r = self._row_coords
        c = self._col_coords

        # Ch 0: species richness
        g[0, 0, r, c] = (env.h > 1).sum(dim=0).float()

        # Ch 1: total population (log-scaled)
        g[0, 1, r, c] = torch.log1p(env.h.sum(dim=0))

        # Ch 2-6: extinction risk per class weighted by species presence
        presence = (env.h > 1).float()  # (n_species, n_cells)
        current_risk = env.current_ext_risk  # (n_species,)
        for k in range(self._n_risk_classes):
            mask = current_risk == k
            if mask.any():
                g[0, 2 + k, r, c] = presence[mask].sum(dim=0)
            else:
                g[0, 2 + k, r, c] = 0.0

        # Ch 7: disturbance
        g[0, 7, r, c] = env.disturbance.data.mean(dim=0)

        # Ch 8: cost
        g[0, 8, r, c] = env.costs.data.mean(dim=0)

        # Ch 9: protection status
        g[0, 9, r, c] = env.protected_cells_mask.float()

        # Ch 10: time
        g[0, 10, r, c] = env._env_step_num / self._time_rescale

        # Ch 11: valid cell mask (already filled in __init__, never changes)

        return g

    def _set_rescaler(self, env: BioEnv) -> None:
        """Compute per-channel normalization stats from initial environment.

        Args:
            env: Environment at initial state.
        """
        grid = self._fill_grid(env)  # (1, 12, H, W)

        # Only compute stats over valid cells to avoid bias from zero-filled areas.
        # Extract valid cells: (12, n_cells)
        valid = grid[0, :, self._row_coords, self._col_coords]  # (12, n_cells)

        self._rescaler_mean = valid.mean(dim=1).view(1, self.cnn.n_input_channels, 1, 1)
        std = valid.std(dim=1)
        std = torch.maximum(std, torch.ones_like(std))
        self._rescaler_std = std.view(1, self.cnn.n_input_channels, 1, 1)

    def _normalize_grid(self, grid: torch.Tensor) -> torch.Tensor:
        """Apply per-channel z-score normalization with clipping.

        Args:
            grid: Raw grid tensor of shape (1, 12, H, W).

        Returns:
            Normalized grid, same shape.
        """
        z = (grid - self._rescaler_mean) / self._rescaler_std
        return torch.clamp(z, -self.CLIP_RANGE, self.CLIP_RANGE)

    def observe(self, env: BioEnv) -> torch.Tensor:
        """Extract CNN features from current environment state.

        Args:
            env: Environment to observe.

        Returns:
            Feature tensor of shape (n_output_features, n_cells).
        """
        grid = self._fill_grid(env)
        grid = self._normalize_grid(grid)
        cnn_out = self.cnn(grid)  # (1, n_output_features, H, W)

        # Extract valid cells
        features = cnn_out[0, :, self._row_coords, self._col_coords]
        return features  # (n_output_features, n_cells)

    def to(self, device: torch.device | str) -> FeatureExtractorCNN:
        """Move extractor to specified device.

        Args:
            device: Target device.

        Returns:
            Self for chaining.
        """
        self.device = torch.device(device)
        self.cnn = self.cnn.to(self.device)
        self._grid = self._grid.to(self.device)
        self._row_coords = self._row_coords.to(self.device)
        self._col_coords = self._col_coords.to(self.device)
        self._rescaler_mean = self._rescaler_mean.to(self.device)
        self._rescaler_std = self._rescaler_std.to(self.device)
        return self
