"""Reward calculation for conservation planning.

This module provides reward functions for evaluating conservation strategies,
including extinction risk changes and protection costs.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from captain.environment.bioenv import BioEnv

logger = logging.getLogger(__name__)


class CalcReward:
    """Base class for reward calculation.

    Subclass this to implement custom reward functions.

    Attributes:
        name: Identifier for this reward component.
        rescaler: Scaling factor applied to reward.
    """

    def __init__(
            self,
            name: str = "base",
            rescaler: float = 1.0,
            positive: bool = True,
    ):
        """Initialize reward calculator.

        Args:
            name: Reward identifier.
            rescaler: Scaling factor (multiplied by reward).
            positive: If False, negate the rescaler.
        """
        self._name = name
        self._rescaler = rescaler if positive else -rescaler
        self._positive = positive
        self._target_achieved = False

    @property
    def name(self) -> str:
        """Reward name."""
        return self._name

    def calc_reward(self, env: BioEnv) -> float:
        """Calculate reward from environment state.

        Args:
            env: Current environment.

        Returns:
            Reward value.
        """
        return 0.0

    def reset(self) -> None:
        """Reset any internal state (called at episode start)."""
        self._target_achieved = False
        pass

    @property
    def name(self) -> str:
        """Reward name."""
        return self._name


class CalcRewardPersistentCost(CalcReward):
    """Reward based on cumulative protection costs.

    Penalizes expensive protection actions by computing the dot product
    of costs and protection levels.
    """

    def __init__(
            self,
            name: str = "cost",
            rescaler: float = 1.0,
            positive: bool = False,
    ):
        """Initialize cost reward.

        Args:
            name: Reward identifier.
            rescaler: Scaling factor (typically 1/total_budget).
            positive: If False, cost is a penalty (default).
        """
        super().__init__(name, rescaler, positive)

    def calc_reward(self, env: BioEnv) -> float:
        """Calculate cost penalty.

        Args:
            env: Current environment.

        Returns:
            Scaled cost penalty (negative if positive=False).
        """
        # Flatten and compute dot product
        costs = env.costs.data.flatten()
        protection = env.protection_matrix.data.flatten()
        reward = torch.dot(costs, protection).item()
        return reward * self._rescaler


class CalcRewardExtRisk(CalcReward):
    """Reward based on changes in species extinction risk.

    Tracks species movement between IUCN threat categories and applies
    weighted rewards/penalties for conservation outcomes.
    """

    def __init__(
            self,
            name: str = "extinction_risk",
            rescaler: float = 1.0,
            positive: bool = True,
            threat_weights: np.ndarray | torch.Tensor | list | None = None,
            device: torch.device | str = "cpu",
    ):
        """Initialize extinction risk reward.

        Args:
            name: Reward identifier.
            rescaler: Scaling factor.
            positive: If True, improvements yield positive reward.
            threat_weights: Weights per threat category [LC, NT, VU, EN, CR].
                           Typically [1, 0, -8, -16, -32] to penalize extinctions.
            device: PyTorch device.

        Raises:
            ValueError: If threat_weights not provided.
        """
        super().__init__(name, rescaler, positive)

        if threat_weights is None:
            raise ValueError(
                "threat_weights must be specified, e.g. [1, 0, -8, -16, -32]"
            )

        self.device = torch.device(device)

        if isinstance(threat_weights, (list, np.ndarray)):
            threat_weights = torch.tensor(threat_weights, dtype=torch.float32)
        self._threat_weights = threat_weights.to(self.device)

        self._previous_status_counts: torch.Tensor | None = None

    def calc_reward(self, env: BioEnv) -> float:
        """Calculate reward from extinction risk changes.

        Args:
            env: Current environment.

        Returns:
            Weighted sum of species shifts between risk categories.
        """
        # Ensure weights are on the same device as env
        if self._threat_weights.device != env.device:
            self._threat_weights = self._threat_weights.to(env.device)

        # Get current counts per risk class
        current_counts = env.ext_risk.species_per_class(env.current_ext_risk)

        # Initialize if first step
        if self._previous_status_counts is None:
            self._previous_status_counts = env.ext_risk._init_status_counts.clone()

        # Ensure previous counts on same device
        if self._previous_status_counts.device != current_counts.device:
            self._previous_status_counts = self._previous_status_counts.to(
                current_counts.device
            )

        # Calculate shift
        diff = current_counts - self._previous_status_counts

        # Apply weights and normalize
        weighted_diff = (diff * self._threat_weights) / env.n_species

        # Update snapshot for next step
        self._previous_status_counts.copy_(current_counts)

        return weighted_diff.sum().item() * self._rescaler

    def reset(self) -> None:
        """Reset tracking state."""
        self._previous_status_counts = None
        self._target_achieved = False

    def to(self, device: torch.device | str) -> CalcRewardExtRisk:
        """Move to specified device.

        Args:
            device: Target device.

        Returns:
            Self for chaining.
        """
        self.device = torch.device(device)
        self._threat_weights = self._threat_weights.to(self.device)
        if self._previous_status_counts is not None:
            self._previous_status_counts = self._previous_status_counts.to(self.device)
        return self


class CalcRewardSpecieValue(CalcReward):
    """Reward based on total amount of species 'value'"""

    def __init__(
            self,
            trait_name: str,
            name: str | None = None,
            rescaler: float = 1.0,
            positive: bool = True,
            device: torch.device | str = "cpu",
            trait_column_indx: int | None = None,  # if None set based on trait_name
            protected_value: bool = False,  # If true only calculate the total value within protected areas
            species_range: bool = False,
            # True: weigh species only based on presence/absence; False: weigh by population
    ):
        """Initialize species value reward.

        Args:
            trait_name: which trait defines species value (from env.trait_map)
            name: Reward identifier.
            rescaler: Scaling factor.
            positive: If True, improvements yield positive reward.
            threat_weights: Weights per threat category [LC, NT, VU, EN, CR].
                           Typically [1, 0, -8, -16, -32] to penalize extinctions.
            device: PyTorch device.

        Raises:
            ValueError: If threat_weights not provided.
        """
        if name is None:
            name = trait_name
        super().__init__(name, rescaler, positive)
        self.device = torch.device(device)
        self._trait_name = trait_name
        self._trait_column_indx = trait_column_indx
        self._protected_value = protected_value
        self._species_range = species_range

    def calc_reward(self, env: BioEnv) -> float:
        if self._trait_column_indx is None:
            self._trait_column_indx = env.trait_map[self._trait_name]

        # average value
        if self._protected_value:
            if self._species_range:
                # protected range per species (i.e. does not weigh more species
                # with higher carrying capacity)
                protected_amount_per_species = env.protected_range / torch.clamp(
                    env.protected_cells_mask.sum(), min=1.0
                )
            else:
                # Result is the average abundance per masked cell for each species
                # (i.e. weighs more species with higher carrying capacity all else equal
                # e.g. carbon or biomass)
                protected_amount_per_species = env.protected_population / torch.clamp(
                    env.protected_cells_mask.sum(), min=1.0
                )

            # Dot product with value (species-specific trait)
            reward = torch.dot(
                env._species_traits[:, self._trait_column_indx],
                protected_amount_per_species,
            )
        else:
            reward = torch.dot(
                env._species_traits[:, self._trait_column_indx], env.h.mean(dim=1)
            )
        return reward.item() * self._rescaler


class CalcRewardDistFromTarget(CalcReward):
    """Reward based on min distance to target, requires env with distance_calculator attribute."""

    def __init__(
            self,
            name: str = "distance_from_target",
            rescaler: float = 1.0,
            positive: bool = False,  # minimize
            exponent: float = 1,
    ):
        """Initialize distance reward.

        Args:
            name: Reward identifier.
            rescaler: Scaling factor (typically 1/total_budget).
            positive: If False, cost is a penalty (default).
        """
        super().__init__(name, rescaler, positive)
        self._exponent = exponent




class CalcRewardEdgeEffect(CalcReward):
    """Reward based on min distance to target, requires env with distance_calculator attribute."""

    def __init__(
            self,
            name: str = "edge_effect",
            rescaler: float = 1.0,
            positive: bool = True,  # maximize
            exponent: float = 1,
    ):
        """Initialize distance reward.

        Args:
            name: Reward identifier.
            rescaler: Scaling factor (typically 1/total_budget).
            positive: If False, cost is a penalty (default).
        """
        super().__init__(name, rescaler, positive)
        self._exponent = exponent

