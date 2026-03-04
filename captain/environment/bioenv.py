"""Biodiversity simulation environment.

This module provides the BioEnv class for simulating species population dynamics
under climate change and conservation interventions.
"""

from __future__ import annotations

import logging
import warnings
from typing import TYPE_CHECKING

import numpy as np
import torch
from scipy import sparse as sp

from captain.data.extinction_risk import ExtinctionRisk
from captain.data.spatial_data import SpatialData
from captain.utils import grid_utils
from captain.utils.grid_utils import scipy_sparse_to_torch

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class BioEnv:
    """Biodiversity simulation environment.

    Simulates species population dynamics including dispersal, growth, and
    mortality under varying climate conditions and protection strategies.

    Attributes:
        n_species: Number of species in simulation.
        n_cells: Number of spatial cells.
        h: Current population matrix of shape (n_species, n_cells).
        device: PyTorch device for computations.

    """

    def __init__(
        self,
        sdms: SpatialData,
        disturbance: SpatialData,
        costs: SpatialData,
        protection_matrix: SpatialData,
        growth_rates: np.ndarray | torch.Tensor,
        sensitivity_rates: np.ndarray | torch.Tensor,
        species_k: np.ndarray | torch.Tensor,
        ext_risk: ExtinctionRisk,
        mortality_rates: float | np.ndarray | torch.Tensor = 1.0,
        dispersal_rates: float | np.ndarray | torch.Tensor = 1.0,
        dispersal_cutoff: int = 3,
        cached_dispersal_matrix: sp.csr_matrix | torch.Tensor | None = None,
        species_traits: np.ndarray | torch.Tensor | None = None,
        device: torch.device | str = "cpu",
    ):
        """Initialize biodiversity environment.

        Args:
            sdms: Species distribution models (habitat suitability).
            disturbance: Disturbance layers (e.g., climate stress).
            costs: Protection cost layers.
            protection_matrix: Current protection levels.
            growth_rates: Population growth rate per species, shape (n_species,).
            sensitivity_rates: Sensitivity to disturbance, shape (n_species, n_disturbance).
            species_k: Carrying capacity per species, shape (n_species,).
            ext_risk: Extinction risk classifier.
            dispersal_rates: Dispersal rate (scalar or per-species array).
            cached_dispersal_matrix: Pre-computed dispersal matrix.
            species_traits: Additional species traits that can be used for feature extraction
                            but do not affect the evolution of the environment
            device: PyTorch device ('cpu', 'cuda', 'mps').
        """
        self.device = torch.device(device)

        # Spatial data
        self.sdms = sdms.to(self.device)
        self.disturbance = disturbance.to(self.device)
        self.protection_matrix = protection_matrix.to(self.device)
        self.costs = costs.to(self.device)

        # Dimensions
        self.n_species = sdms.shape[0]
        self.n_cells = sdms.data.shape[1]

        # Life-history traits - convert to tensors
        self._growth_rates = self._to_tensor(growth_rates)
        self._species_sensitivity = self._to_tensor(sensitivity_rates)
        self._species_k = self._to_tensor(species_k)
        self._species_traits = (
            self._to_tensor(species_traits) if species_traits else None
        )

        # Handle dispersal rates
        if isinstance(dispersal_rates, (int, float)):
            self._lambda_0 = float(dispersal_rates)
            self._is_scalar_dispersal = True
        else:
            self._lambda_0 = self._to_tensor(dispersal_rates)
            self._is_scalar_dispersal = False
        self._dispersal_cutoff = dispersal_cutoff

        # Set maximum mortality rate

        np.mean(sensitivity_rates, axis=0)

        if isinstance(mortality_rates, (int, float)):
            self._death_rates = self._to_tensor(
                np.repeat(mortality_rates, self.n_species)
            )
        else:
            self._death_rates = self._to_tensor(mortality_rates)

        # Extinction risk classifier
        self.ext_risk = ext_risk.to(self.device)

        # Step counters
        self._env_step_num = 0
        self._step_num = 0

        # Setup dispersal matrix
        # Strategy: Use SciPy sparse for CPU (10x faster), PyTorch sparse for CUDA
        self._use_scipy_sparse = self.device.type in ("cpu", "mps")

        if cached_dispersal_matrix is not None:
            if isinstance(cached_dispersal_matrix, sp.spmatrix):
                self._scipy_dist = cached_dispersal_matrix.tocsr()
                if not self._use_scipy_sparse:
                    self._cached_dist = scipy_sparse_to_torch(
                        cached_dispersal_matrix, self.device
                    )
            else:
                self._cached_dist = cached_dispersal_matrix.to(self.device)
                self._use_scipy_sparse = False
        elif self._is_scalar_dispersal:
            self._scipy_dist = grid_utils.dispersal_distances_threshold_coords(
                self._lambda_0, self.sdms._coords, threshold=self._dispersal_cutoff
            )
            if not self._use_scipy_sparse:
                self._cached_dist = scipy_sparse_to_torch(self._scipy_dist, self.device)
        else:
            # Per-species dispersal rates - pre-compute all species matrices
            # This trades init time for much faster step time
            self._species_dist = self._build_species_dispersal_matrices()

        # Validate growth rates
        if self._growth_rates.min() < 1.0:
            warnings.warn("Growth rates should be >= 1.0", UserWarning, stacklevel=2)

        # Initialize state
        self.reset()

    def _to_tensor(self, arr: np.ndarray | torch.Tensor) -> torch.Tensor:
        """Convert array to tensor on device."""
        if isinstance(arr, np.ndarray):
            return torch.from_numpy(arr.astype(np.float32)).to(self.device)
        return arr.to(self.device, dtype=torch.float32)

    def _build_species_dispersal_matrices(self) -> list:
        """Pre-compute species-specific dispersal matrices.

        For per-species dispersal rates, we pre-compute D_s = D_base^(1/lambda_s)
        for each species at init time. This trades initialization time for
        much faster stepping (sparse matmul vs dense power operation).

        Returns:
            List of sparse matrices (SciPy for CPU, PyTorch for CUDA).
        """
        logger.info(f"Pre-computing {self.n_species} species dispersal matrices...")

        # Compute base matrix with lambda=1
        scipy_base = grid_utils.dispersal_distances_threshold_coords(
            1.0, self.sdms._coords, threshold=self._dispersal_cutoff
        ).tocsr()

        # Compute inverse lambdas
        inv_lambdas = 1.0 / self._lambda_0  # Already a tensor

        if self._use_scipy_sparse:
            # CPU/MPS: Build SciPy sparse matrices (much faster)
            self._species_scipy_dist = []
            for s in range(self.n_species):
                # Apply power to values only (keeps same sparsity pattern)
                new_data = scipy_base.data ** inv_lambdas[s].item()
                sparse_s = sp.csr_matrix(
                    (new_data, scipy_base.indices, scipy_base.indptr),
                    shape=scipy_base.shape,
                )
                self._species_scipy_dist.append(sparse_s)
            logger.info(
                f"Pre-computed {len(self._species_scipy_dist)} SciPy dispersal matrices"
            )
            return []  # Not using PyTorch sparse for CPU
        else:
            # CUDA: Build PyTorch sparse tensors
            crow_indices = torch.from_numpy(scipy_base.indptr.astype(np.int64))
            col_indices = torch.from_numpy(scipy_base.indices.astype(np.int64))
            base_values = torch.from_numpy(scipy_base.data.astype(np.float32))
            shape = scipy_base.shape

            species_dist = []
            for s in range(self.n_species):
                species_values = base_values ** inv_lambdas[s].item()
                sparse_s = torch.sparse_csr_tensor(
                    crow_indices.clone(),
                    col_indices.clone(),
                    species_values,
                    size=shape,
                    device=self.device,
                    dtype=torch.float32,
                )
                species_dist.append(sparse_s)

            logger.info(f"Pre-computed {len(species_dist)} PyTorch dispersal matrices")
            return species_dist

    def env_step(self) -> None:
        """Execute one environment step: dispersal, growth, and death"""
        # 1. Calculate the mask before anything changes
        below_k = self._h <= self._current_carrying_capacity

        # 2. DISPERSAL (Computed for all, then applied selectively)
        if self._is_scalar_dispersal:
            if self._use_scipy_sparse:
                h_np = self._h.cpu().numpy()
                dispersed_h = torch.from_numpy(h_np @ self._scipy_dist.T).to(
                    self.device
                )
            else:
                # We compute the potential new state for everyone
                dispersed_h = torch.sparse.mm(self._cached_dist, self._h.t()).t()
        else:
            # Per-species dispersal with pre-computed sparse matrices
            if self._use_scipy_sparse:
                # CPU/MPS path: Use SciPy sparse
                dispersed_h = self._h.cpu().numpy()
                for s in range(self.n_species):
                    dispersed_h[s] = dispersed_h[s] @ self._species_scipy_dist[s].T
                dispersed_h = torch.from_numpy(dispersed_h).to(
                    self.device, dtype=torch.float32
                )
            else:
                # CUDA path: Use PyTorch sparse on GPU
                dispersed_h = self._h.clone()
                for s in range(self.n_species):
                    dispersed_h[s] = torch.sparse.mm(
                        self._species_dist[s].t(), self._h[s : s + 1].t()
                    ).squeeze()

        # Only update the cells that were below K
        self._h[below_k] = dispersed_h[below_k]

        # 2. GROWTH & CLIPPING (Fused)
        # We use a boolean mask once to avoid redundant scans

        # Growth: In-place multiplication only where below K
        # Using Option 1 from our previous discussion for memory efficiency
        idx_s, idx_x = torch.where(below_k)
        self._h[idx_s, idx_x] *= self._growth_rates[idx_s]

        # Carrying capacity clip: In-place
        # clamp_max_ is faster and uses zero extra memory
        self._h[idx_s, idx_x] = torch.clamp_max(
            self._h[idx_s, idx_x], self._current_carrying_capacity[idx_s, idx_x]
        )

        # 3. DEATH (Non-linear mortality for cells above K)
        # Death rates (only for cells above K e.g. due to suitability or disturbance changes
        above_k = ~below_k
        if above_k.any():
            idx_s_a, idx_x_a = torch.where(above_k)

            h_sub = self._h[idx_s_a, idx_x_a]
            cap_sub = self._current_carrying_capacity[idx_s_a, idx_x_a]

            # Pre-calculate difference to reuse
            diff = h_sub - cap_sub

            # Calculate mortality proportion: death_rate * (1 - exp(1 - h/K))
            # h = h - (death_rate * (1 - exp(1 - h / K))) * (h - K)
            # where: (death_rate * (1 - exp(1 - h / K))) is the proportion of
            # offshooting individuals that die
            term = 1.0 - h_sub / cap_sub
            term.exp_()
            prop_death = self._death_rates[idx_s_a] * (1.0 - term)
            # Final update: h = h - (prop_death * diff)
            self._h[idx_s_a, idx_x_a] -= prop_death * diff

        self._env_step_num += 1

    def step(self, env_step: bool = True) -> None:
        """Advance simulation by one timestep.

        Args:
            env_step: If True, run full environment step (dispersal, growth).
                     If False, only update carrying capacity.
        """
        self.update_carrying_capacity()
        if env_step:
            self.env_step()
            self.disturbance.update()
            self.costs.update()
            self.sdms.update()
        self._step_num += 1

    def update_protection_matrix(self, idx: torch.Tensor | np.ndarray | list) -> None:
        """Add protection to specified cells.

        Args:
            idx: Indices of cells to protect.
        """
        self.protection_matrix.update_col_values(idx, 1.0)

    def reset(self) -> None:
        """Reset environment to initial state."""
        self._env_step_num = 0
        self._step_num = 0
        self.disturbance.reset()
        self.protection_matrix.reset()
        self.costs.reset()
        self.sdms.reset()
        self.update_carrying_capacity()
        self.init_h()
        self.set_init_ext_risk()

    def init_h(self) -> None:
        """Initialize population to carrying capacity."""
        self.set_h(self._current_carrying_capacity.clone())

    def set_h(self, h: torch.Tensor) -> None:
        """Set population matrix.

        Args:
            h: New population matrix of shape (n_species, n_cells).
        """
        if isinstance(h, np.ndarray):
            h = torch.from_numpy(h.astype(np.float32))
        self._h = h.to(self.device, dtype=torch.float32)

    def update_carrying_capacity(self) -> None:
        """Update carrying capacity based on disturbance and protection."""
        # 1. Compute effective disturbance: (d, x)
        # mean disturbance effect (if multiple disturbance layers)
        eff_dist = 1.0 - self.protection_matrix.data
        eff_dist = eff_dist * self.disturbance.data
        eff_dist = eff_dist / self.disturbance.shape[0]

        # 2. Project disturbance to species level: (s, d) @ (d, x) -> (s, x)
        k_dist = self._species_sensitivity @ eff_dist

        # 3. Transform: k_dist = (1 - impact)
        k_dist = 1.0 - k_dist

        # 4. Apply SDM and species carrying capacity
        # data_min_threshold returns 0 where suitability is below
        # a predefined threshold (sdms._min_threshold_per_channel)
        k_dist = k_dist * self.sdms.data_min_threshold
        k_dist = k_dist * self._species_k.unsqueeze(1)

        self._current_carrying_capacity = k_dist

    def set_init_ext_risk(self) -> None:
        """Initialize extinction risk baseline from current state."""
        self.ext_risk.set_init_values(
            init_pop=self._h.sum(dim=1),
            init_protected_pop=self.protected_population,
        )

    @property
    def h(self) -> torch.Tensor:
        """Current population matrix of shape (n_species, n_cells)."""
        return self._h

    @property
    def reconstruct_h_grid(self) -> np.ndarray:
        """Reconstruct population as 3D grid for visualization.

        Returns:
            NumPy array of shape (n_species, x, y) with NaN for masked cells.
        """
        return grid_utils.reconstruct_grid(
            self._h.cpu().numpy(),
            self.sdms._coords,
            self.sdms.shape[1:],
        )

    @property
    def protected_cells_mask(self) -> torch.Tensor:
        """Boolean mask of protected cells, shape (n_cells,)."""
        return self.protection_matrix._nonzero_cells_mask

    @property
    def protected_population(self) -> torch.Tensor:
        """Total population in protected areas per species, shape (n_species,)."""
        # (s, x) @ (x,) -> (s,)
        return self._h @ self.protected_cells_mask.float()

    @property
    def current_ext_risk(self) -> torch.Tensor:
        """Current extinction risk per species."""
        return self.ext_risk.classify(
            self._h.sum(dim=1),
            self.protected_population,
        )

    @property
    def protected_population_fraction(self) -> torch.Tensor:
        """Fraction of population in protected areas per species."""
        return self.protected_population / self._h.sum(dim=1)

    def to(self, device: torch.device | str) -> BioEnv:
        """Move environment to specified device.

        Args:
            device: Target device.

        Returns:
            Self for chaining.
        """
        self.device = torch.device(device)
        self._h = self._h.to(self.device)
        self._growth_rates = self._growth_rates.to(self.device)
        self._species_sensitivity = self._species_sensitivity.to(self.device)
        self._species_k = self._species_k.to(self.device)
        self._current_carrying_capacity = self._current_carrying_capacity.to(
            self.device
        )

        if hasattr(self, "_cached_dist"):
            self._cached_dist = self._cached_dist.to(self.device)
        if hasattr(self, "_species_dist"):
            # Note: For MPS, keep sparse matrices on CPU
            sparse_device = "cpu" if self.device.type == "mps" else self.device
            self._species_dist = [d.to(sparse_device) for d in self._species_dist]
        if hasattr(self, "_inv_lambdas") and isinstance(
            self._inv_lambdas, torch.Tensor
        ):
            self._inv_lambdas = self._inv_lambdas.to(self.device)

        self.sdms = self.sdms.to(self.device)
        self.disturbance = self.disturbance.to(self.device)
        self.costs = self.costs.to(self.device)
        self.protection_matrix = self.protection_matrix.to(self.device)
        self.ext_risk = self.ext_risk.to(self.device)

        return self
