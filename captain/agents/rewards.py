"""Reward calculation for conservation planning.

This module provides reward functions for evaluating conservation strategies,
including extinction risk changes and protection costs.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import torch

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
        pass


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


class Rewards:
    """Aggregator for multiple reward components.

    Combines multiple CalcReward instances and computes weighted sum.

    Attributes:
        episode_rewards: Cumulative rewards per component.
        episode_reward_history: Per-step reward history.

    Example:
        >>> rewards = Rewards([
        ...     CalcRewardExtRisk(threat_weights=[1, 0, -8, -16, -32]),
        ...     CalcRewardPersistentCost(rescaler=0.1),
        ... ])
        >>> rewards.reset()
        >>> total = rewards.calc_total_reward(env)
    """

    def __init__(
            self,
            reward_obj_list: list[CalcReward] | None,
            discount_factor: float = 1.0,
            reward_weights: np.ndarray | torch.Tensor | list | None = None,
            cumulative_reward: bool = True,
    ):
        """Initialize reward aggregator.

        Args:
            reward_obj_list: List of CalcReward instances.
            discount_factor: Discount for future rewards (unused currently).
            reward_weights: Weights for combining rewards (default: uniform).
            cumulative_reward: If True, accumulate rewards across steps.
        """
        self._reward_obj_list = [] if reward_weights is None else list(reward_obj_list)
        self._discount_factor = discount_factor
        self._cumulative_reward = cumulative_reward

        if reward_weights is None:
            self._reward_weights = torch.ones(len(self._reward_obj_list))
        elif isinstance(reward_weights, (list, np.ndarray)):
            self._reward_weights = torch.tensor(reward_weights, dtype=torch.float32)
        else:
            self._reward_weights = reward_weights

        self.reset()

    def calc_reward(self, env: BioEnv) -> None:
        """Calculate rewards for current step.

        Args:
            env: Current environment.
        """
        rewards = [obj.calc_reward(env) for obj in self._reward_obj_list]
        reward_tensor = torch.tensor(rewards, dtype=torch.float32)
        self.episode_rewards.add_(reward_tensor)
        self.episode_reward_history.append(rewards)

    def reset(self) -> None:
        """Reset for new episode."""
        self.episode_rewards = torch.zeros(len(self._reward_obj_list))
        self.episode_reward_history: list[list[float]] = []
        for obj in self._reward_obj_list:
            obj.reset()

    def calc_total_reward(self, env: BioEnv) -> float:
        """Calculate and return weighted total reward.

        Args:
            env: Current environment.

        Returns:
            Weighted sum of all reward components.
        """
        self.calc_reward(env)
        return self.get_weighted_reward()

    def get_weighted_reward(self) -> float:
        """Get current weighted total reward.

        Returns:
            Weighted sum of episode rewards.
        """
        return (self.episode_rewards * self._reward_weights).sum().item()

    @property
    def names(self) -> list[str]:
        """Names of all reward components."""
        return [obj.name for obj in self._reward_obj_list]


class NoRewards(Rewards):
    def __init__(self, reward_obj_list=None):
        super().__init__(reward_obj_list)
        self.reset()

    def calc_reward(self, env: BioEnv) -> None:
        pass

    def reset(self) -> None:
        """Reset for new episode."""
        self.episode_rewards = torch.zeros(1)
        self.episode_reward_history: list[list[float]] = []

    def calc_total_reward(self, env: BioEnv) -> float:
        return self.get_weighted_reward()

    def get_weighted_reward(self) -> float:
        return 0

    @property
    def names(self) -> list[str]:
        return ["no_reward"]
