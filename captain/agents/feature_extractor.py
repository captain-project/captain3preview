"""Feature extraction from environment state for policy network input.

This module provides the FeatureExtractor class that transforms raw environment
state into normalized features suitable for neural network input.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from captain.utils import grid_utils
from captain.utils.grid_utils import scipy_sparse_to_torch

if TYPE_CHECKING:
    from captain.environment.bioenv import BioEnv

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """Extract and normalize features from environment state.

    Transforms raw environment observations into normalized feature vectors
    suitable for policy network input. Features include time, disturbance,
    population metrics, extinction risk, costs, and protection status.

    Attributes:
        n_features: Total number of features.
        feature_names: List of feature names.
        device: PyTorch device.

    Example:
        >>> extractor = FeatureExtractor(env, device="cuda")
        >>> features = extractor.observe(env)  # shape: (n_features, n_cells)
        >>> actions = policy(features)
    """

    # Feature constants
    CLIP_RANGE: float = 5.0
    DEFAULT_TIME_RESCALE: float = 10.0

    def __init__(
        self,
        env: BioEnv,
        feature_set: list[str] | None = None,
        convolution: int = 5,
        keys_to_reset: list[str] | None = None,
        time_rescale: float = DEFAULT_TIME_RESCALE,
        device: torch.device | str | None = None,
    ):
        """Initialize feature extractor.

        Args:
            env: Environment to extract features from.
            feature_set: List of features to extract (default: all).
            convolution: Neighborhood radius for convolution features.
            keys_to_reset: Features that should not be z-score normalized.
            time_rescale: Divisor for time normalization.
            device: PyTorch device (default: same as env).
        """
        self.device = torch.device(device) if device else env.device

        # Feature configuration
        if feature_set is None:
            self._feature_set = [
                "time",
                "disturbance",
                "disturbance_conv",
                "species_richness",
                "total_population",
                "current_ext_risk",
                "cost",
                "protection_matrix",
                "protection_matrix_conv",
            ]
        else:
            self._feature_set = list(feature_set)

        # Build feature name list (expand current_ext_risk to per-class features)
        self._feature_names = list(self._feature_set)
        if "current_ext_risk" in self._feature_set:
            risk_names = [f"ext_risk_{i}" for i in range(env.ext_risk._n_classes)]
            idx = self._feature_set.index("current_ext_risk")
            self._feature_names[idx : idx + 1] = risk_names
            self._n_risk_classes = env.ext_risk._n_classes
        else:
            self._n_risk_classes = 0

        # Feature map for quick lookup
        self.feature_map = {name: i for i, name in enumerate(self._feature_names)}
        self._n_features = len(self._feature_names)

        # Convolution matrix for neighborhood features
        self.convolution = convolution
        scipy_conv = grid_utils.compute_convolution_matrix(
            env.disturbance._coords, radius=int(self.convolution - 1 / 2)
        )
        self._conv_matrix = scipy_sparse_to_torch(scipy_conv, self.device)

        # Feature buffer
        self.features = torch.zeros(
            (self._n_features, env.n_cells),
            device=self.device,
            dtype=torch.float32,
        )

        # Rescaling configuration
        if keys_to_reset is None:
            self.keys_to_reset = [
                "time",
                "disturbance",
                "disturbance_conv",
                "protection_matrix",
                "protection_matrix_conv",
            ]
        else:
            self.keys_to_reset = list(keys_to_reset)

        self._features_to_reset = [
            self.feature_map[k] for k in self.keys_to_reset if k in self.feature_map
        ]

        self._time_rescale = time_rescale

        # Initialize rescaler
        self.set_rescaler(env)

    @property
    def n_features(self) -> int:
        """Number of features."""
        return self._n_features

    def extract_features(self, env: BioEnv) -> torch.Tensor:
        """Extract raw features from environment.

        Args:
            env: Environment to extract features from.

        Returns:
            Feature tensor of shape (n_features, n_cells).
        """
        obs = self.features

        if "time" in self._feature_set:
            idx = self.feature_map["time"]
            obs[idx].fill_(env._env_step_num / self._time_rescale)

        if "disturbance" in self._feature_set:
            idx = self.feature_map["disturbance"]
            obs[idx] = env.disturbance.data.mean(dim=0)

        if "disturbance_conv" in self._feature_set:
            idx = self.feature_map["disturbance_conv"]
            disturbance_mean = env.disturbance.data.mean(dim=0)
            # Sparse matrix multiplication: (n_cells,) @ sparse(n_cells, n_cells)
            # Note: For MPS, sparse matrix is on CPU
            sparse_device = self._conv_matrix.device
            if sparse_device != self.device:
                # MPS path: transfer to CPU for sparse op, then back
                disturbance_cpu = disturbance_mean.to(sparse_device)
                result = torch.sparse.mm(
                    self._conv_matrix, disturbance_cpu.unsqueeze(1)
                ).squeeze(1)
                obs[idx] = result.to(self.device)
            else:
                obs[idx] = torch.sparse.mm(
                    self._conv_matrix, disturbance_mean.unsqueeze(1)
                ).squeeze(1)

        if "species_richness" in self._feature_set:
            idx = self.feature_map["species_richness"]
            obs[idx] = (env.h > 1).sum(dim=0).float()

        if "total_population" in self._feature_set:
            idx = self.feature_map["total_population"]
            pop_sum = env.h.sum(dim=0)
            obs[idx] = torch.log1p(pop_sum)

        if "protection_matrix" in self._feature_set:
            idx = self.feature_map["protection_matrix"]
            obs[idx] = env.protected_cells_mask.float()

        if "protection_matrix_conv" in self._feature_set:
            idx = self.feature_map["protection_matrix_conv"]
            protection_m = env.protected_cells_mask.float()
            # Sparse matrix multiplication: (n_cells,) @ sparse(n_cells, n_cells)
            # Note: For MPS, sparse matrix is on CPU
            sparse_device = self._conv_matrix.device
            if sparse_device != self.device:
                # MPS path: transfer to CPU for sparse op, then back
                disturbance_cpu = protection_m.to(sparse_device)
                result = torch.sparse.mm(
                    self._conv_matrix, disturbance_cpu.unsqueeze(1)
                ).squeeze(1)
                obs[idx] = result.to(self.device)
            else:
                obs[idx] = torch.sparse.mm(
                    self._conv_matrix, protection_m.unsqueeze(1)
                ).squeeze(1)

        if "current_ext_risk" in self._feature_set:
            start = self.feature_map["ext_risk_0"]
            end = start + self._n_risk_classes
            risk_view = obs[start:end]
            risk_view.zero_()

            # One-hot encoding of risk per cell weighted by species presence
            current_risk = env.current_ext_risk  # (n_species,)
            presence = (env.h > 1).float()  # (n_species, n_cells)

            # Scatter add: for each species, add its presence to its risk class row
            for risk_class in range(self._n_risk_classes):
                mask = current_risk == risk_class
                if mask.any():
                    risk_view[risk_class] = presence[mask].sum(dim=0)

        if "cost" in self._feature_set:
            idx = self.feature_map["cost"]
            obs[idx] = env.costs.data.mean(dim=0)

        return obs

    def observe(self, env: BioEnv) -> torch.Tensor:
        """Extract and normalize features from environment.

        Args:
            env: Environment to observe.

        Returns:
            Normalized feature tensor of shape (n_features, n_cells).
        """
        raw_obs = self.extract_features(env)
        return self.transform(raw_obs)

    def transform(self, obs: torch.Tensor) -> torch.Tensor:
        """Apply z-score normalization with outlier clipping.

        Args:
            obs: Raw observation tensor.

        Returns:
            Normalized tensor clipped to [-CLIP_RANGE, CLIP_RANGE].
        """
        z = (obs - self._rescaler_mean) / self._rescaler_std
        return torch.clamp(z, -self.CLIP_RANGE, self.CLIP_RANGE)

    def set_rescaler(self, env: BioEnv) -> None:
        """Compute normalization statistics from current environment state.

        Args:
            env: Environment to compute statistics from.
        """
        obs = self.extract_features(env)

        # Compute mean and std per feature
        self._rescaler_std = torch.std(obs, dim=1, keepdim=True)
        self._rescaler_std = torch.maximum(
            self._rescaler_std,
            torch.ones_like(self._rescaler_std),
        )
        self._rescaler_mean = torch.mean(obs, dim=1, keepdim=True)

        # Don't rescale certain features
        self._rescaler_std[self._features_to_reset] = 1.0
        self._rescaler_mean[self._features_to_reset] = 0.0

    def plot_features(
        self, env: BioEnv, rescale: bool = True, outdir: str | Path | None = None
    ) -> None:
        """Plot all features as spatial grids.

        Args:
            env: Environment to visualize.
            rescale: If True, show normalized features.
        """
        from captain.utils import plots

        obs = self.observe(env) if rescale else self.extract_features(env)

        obs_np = obs.cpu().numpy()
        feature_3d = grid_utils.reconstruct_grid(
            obs_np, env.disturbance._coords, env.disturbance._data_shape[1:]
        )
        for i in range(self._n_features):
            plots.plot_grid(
                feature_3d[i],
                title=self._feature_names[i],
                outfile=os.path.join(str(outdir), f"f_{self._feature_names[i]}.png"),
            )

    def to(self, device: torch.device | str) -> FeatureExtractor:
        """Move extractor to specified device.

        Args:
            device: Target device.

        Returns:
            Self for chaining.
        """
        self.device = torch.device(device)
        self.features = self.features.to(self.device)
        self._conv_matrix = self._conv_matrix.to(self.device)
        self._rescaler_std = self._rescaler_std.to(self.device)
        self._rescaler_mean = self._rescaler_mean.to(self.device)
        return self
