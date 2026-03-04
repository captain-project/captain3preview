"""Extinction risk classification system based on IUCN categories.

This module provides classes for classifying species extinction risk based on
population decline and protection levels, following IUCN Red List methodology.
"""

from __future__ import annotations

import logging

import numpy as np
import torch

logger = logging.getLogger(__name__)


class ExtinctionRisk:
    """Dynamic extinction risk classifier based on population and protection.

    Classifies species into risk categories (0=Least Concern to n_classes-1=Critical)
    based on population decline from initial values and proportion of population
    in protected areas.

    Attributes:
        init_status: Initial risk status per species.
        init_pop: Initial population per species (set via set_init_values).
        n_classes: Number of risk categories.

    Example:
        >>> ext_risk = ExtinctionRisk(
        ...     init_status=torch.zeros(10, dtype=torch.long),
        ...     n_classes=5
        ... )
        >>> ext_risk.set_init_values(init_pop, init_protected_pop)
        >>> current_risk = ext_risk.classify(current_pop, protected_pop)
    """

    def __init__(
        self,
        init_status: np.ndarray | torch.Tensor,
        n_classes: int = 5,
        init_range: np.ndarray | torch.Tensor | None = None,
        init_pop: np.ndarray | torch.Tensor | None = None,
        init_protected_range: np.ndarray | torch.Tensor | None = None,
        init_protected_pop: np.ndarray | torch.Tensor | None = None,
        alpha: float = 1.0,
        loss_thresholds: np.ndarray | torch.Tensor | None = None,
        protect_thresholds: np.ndarray | torch.Tensor | None = None,
        device: torch.device | str = "cpu",
    ):
        """Initialize extinction risk classifier.

        Args:
            init_status: Initial risk status per species, shape (n_species,).
            n_classes: Number of risk categories (default 5: LC, NT, VU, EN, CR).
            init_range: Initial range size per species.
            init_pop: Initial population per species.
            init_protected_range: Initial protected range per species.
            init_protected_pop: Initial protected population per species.
            alpha: Threshold distribution parameter (1.0 = uniform).
            loss_thresholds: Custom thresholds for population loss.
            protect_thresholds: Custom thresholds for protection benefit.
            device: PyTorch device.
        """
        self.device = torch.device(device)
        self._n_classes = n_classes
        self._alpha = alpha

        # Convert inputs to torch tensors
        if isinstance(init_status, np.ndarray):
            init_status = torch.from_numpy(init_status)
        self._init_status = init_status.to(self.device, dtype=torch.long)

        # Handle optional arrays
        self._init_range = self._to_tensor(init_range)
        self._init_pop = self._to_tensor(init_pop)
        self._init_protected_range = self._to_tensor(init_protected_range)
        self._init_protected_pop = self._to_tensor(init_protected_pop)

        # Setup thresholds
        if loss_thresholds is not None:
            if isinstance(loss_thresholds, np.ndarray):
                loss_thresholds = torch.from_numpy(loss_thresholds)
            self._loss_thresholds = loss_thresholds.to(self.device, dtype=torch.float32)
            if len(self._loss_thresholds) - 1 != self._n_classes:
                raise ValueError(
                    f"Number of loss thresholds ({len(self._loss_thresholds)}) "
                    f"must be n_classes + 1 ({self._n_classes + 1})"
                )
        else:
            beta = torch.arange(self._n_classes + 1, dtype=torch.float32) / self._n_classes
            self._loss_thresholds = (beta ** (1.0 / self._alpha)).to(self.device)

        if protect_thresholds is not None:
            if isinstance(protect_thresholds, np.ndarray):
                protect_thresholds = torch.from_numpy(protect_thresholds)
            self._protect_thresholds = protect_thresholds.to(self.device, dtype=torch.float32)
            if len(self._protect_thresholds) - 1 != self._n_classes:
                raise ValueError(
                    f"Number of protect thresholds ({len(self._protect_thresholds)}) "
                    f"must be n_classes + 1 ({self._n_classes + 1})"
                )
        else:
            beta = torch.arange(self._n_classes + 1, dtype=torch.float32) / self._n_classes
            self._protect_thresholds = (beta ** (1.0 / self._alpha)).to(self.device)

        self._delta_z = torch.zeros_like(self._init_status)
        self._init_status_counts = self.species_per_class(self._init_status)

    def _to_tensor(self, arr: np.ndarray | torch.Tensor | None) -> torch.Tensor | None:
        """Convert array to tensor if not None."""
        if arr is None:
            return None
        if isinstance(arr, np.ndarray):
            return torch.from_numpy(arr.astype(np.float32)).to(self.device)
        return arr.to(self.device, dtype=torch.float32)

    def classify(
        self, current_pop: torch.Tensor, protected_pop: torch.Tensor
    ) -> torch.Tensor:
        """Classify species extinction risk.

        Args:
            current_pop: Current total population per species, shape (n_species,).
            protected_pop: Population in protected areas per species, shape (n_species,).

        Returns:
            Risk classification per species, 0 (lowest) to n_classes-1 (highest).
        """
        # Ensure inputs are on correct device
        current_pop = current_pop.to(self.device, dtype=torch.float32)
        protected_pop = protected_pop.to(self.device, dtype=torch.float32)

        # 1. Standardize Decline: 0 (no loss) to 1 (extinct)
        decline_ratio = torch.clamp(1 - (current_pop / self._init_pop), 0, 1)

        # 2. Protection Ratio: 0 (none) to 1 (full)
        protected_ratio = torch.clamp(
            protected_pop / torch.clamp(current_pop, min=1), 0, 1
        )

        # 3. Calculate Base Risk Increase from decline
        # torch.bucketize is equivalent to np.digitize
        # We use the thresholds excluding first element (0)
        loss_impact = torch.bucketize(decline_ratio, self._loss_thresholds[1:])

        # 4. Calculate Risk Mitigation from protection
        protect_impact = torch.bucketize(protected_ratio, self._protect_thresholds[1:])

        # Result = Initial Status + Impact from Loss - Benefit from Protection
        new_status = self._init_status + loss_impact - protect_impact

        return torch.clamp(new_status, 0, self._n_classes - 1)

    def set_init_values(
        self,
        init_pop: torch.Tensor | np.ndarray,
        init_protected_pop: torch.Tensor | np.ndarray,
        init_protected_range: torch.Tensor | np.ndarray | None = None,
        init_protected_pop_range: torch.Tensor | np.ndarray | None = None,
    ) -> None:
        """Set initial population values for classification baseline.

        Args:
            init_pop: Initial total population per species.
            init_protected_pop: Initial protected population per species.
            init_protected_range: Initial protected range per species.
            init_protected_pop_range: Initial protected population range.
        """
        self._init_pop = self._to_tensor(init_pop)
        self._init_protected_pop = self._to_tensor(init_protected_pop)
        self._init_protected_range = self._to_tensor(init_protected_range)
        self._init_protected_pop_range = self._to_tensor(init_protected_pop_range)

    def species_per_class(
        self, risk_labels: torch.Tensor | None = None, normalize: bool = False
    ) -> torch.Tensor:
        """Count species in each risk class.

        Args:
            risk_labels: Risk classification per species.
            normalize: If True, return proportions instead of counts.

        Returns:
            Count (or proportion) per risk class, shape (n_classes,).
        """
        if risk_labels is None:
            risk_labels = self._init_status

        # Ensure on correct device
        risk_labels = risk_labels.to(self.device)

        # Count occurrences of each class
        counts = torch.zeros(self._n_classes, dtype=torch.float32, device=self.device)
        labels, label_counts = torch.unique(risk_labels, return_counts=True)
        counts[labels.long()] = label_counts.float()

        if normalize:
            counts = counts / counts.sum()

        return counts

    def to(self, device: torch.device | str) -> ExtinctionRisk:
        """Move all tensors to specified device.

        Args:
            device: Target device.

        Returns:
            Self for chaining.
        """
        self.device = torch.device(device)
        self._init_status = self._init_status.to(self.device)
        self._loss_thresholds = self._loss_thresholds.to(self.device)
        self._protect_thresholds = self._protect_thresholds.to(self.device)
        self._delta_z = self._delta_z.to(self.device)
        self._init_status_counts = self._init_status_counts.to(self.device)
        if self._init_pop is not None:
            self._init_pop = self._init_pop.to(self.device)
        if self._init_protected_pop is not None:
            self._init_protected_pop = self._init_protected_pop.to(self.device)
        if self._init_range is not None:
            self._init_range = self._init_range.to(self.device)
        if self._init_protected_range is not None:
            self._init_protected_range = self._init_protected_range.to(self.device)
        return self

    @property
    def init_status(self) -> torch.Tensor:
        """Initial risk status per species."""
        return self._init_status

    @property
    def init_pop(self) -> torch.Tensor | None:
        """Initial population per species."""
        return self._init_pop

    @property
    def delta_z(self) -> torch.Tensor:
        """Change in risk status."""
        return self._delta_z


class ExtinctionRiskStatic(ExtinctionRisk):
    """Static extinction risk that never changes from initial status.

    Useful for scenarios where species risk is externally determined
    and should not change based on simulation dynamics.
    """

    def __init__(
        self,
        init_status: np.ndarray | torch.Tensor,
        n_classes: int = 5,
        device: torch.device | str = "cpu",
    ):
        """Initialize static extinction risk.

        Args:
            init_status: Fixed risk status per species.
            n_classes: Number of risk categories.
            device: PyTorch device.
        """
        super().__init__(init_status=init_status, n_classes=n_classes, device=device)

    def classify(
        self, current_pop: torch.Tensor, protected_pop: torch.Tensor
    ) -> torch.Tensor:
        """Return fixed initial status (ignores population data).

        Args:
            current_pop: Ignored.
            protected_pop: Ignored.

        Returns:
            Initial risk status (unchanged).
        """
        return self._init_status
