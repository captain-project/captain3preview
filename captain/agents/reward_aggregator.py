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
    from captain.agents.rewards import CalcReward
    from captain.environment.bioenv import BioEnv

logger = logging.getLogger(__name__)


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
            reward_calibration: np.ndarray | torch.Tensor | list | None = None,
            cumulative_reward: bool = True,
    ):
        """Initialize reward aggregator.

        Args:
            reward_obj_list: List of CalcReward instances.
            discount_factor: Discount for future rewards (unused currently).
            reward_weights: Weights for combining rewards (default: uniform).
            cumulative_reward: If True, accumulate rewards across steps.
        """
        self._reward_obj_list = [] if reward_obj_list is None else list(reward_obj_list)
        self._discount_factor = discount_factor
        self._cumulative_reward = cumulative_reward

        if reward_weights is None:
            self._reward_weights = torch.ones(len(self._reward_obj_list))
        elif isinstance(reward_weights, (list, np.ndarray)):
            self._reward_weights = torch.tensor(reward_weights, dtype=torch.float32)
        else:
            self._reward_weights = reward_weights

        if reward_calibration is None:
            self._reward_calibration = torch.ones(len(self._reward_obj_list))
        elif isinstance(reward_calibration, (list, np.ndarray)):
            self._reward_calibration = torch.tensor(
                reward_calibration, dtype=torch.float32
            )
        else:
            self._reward_calibration = reward_calibration

        self._reward_initial_weights = self._reward_weights.clone()

        self.reset()

    def calc_reward(self, env: BioEnv) -> None:
        """Calculate rewards for current step.

        Args:
            env: Current environment.
        """
        rewards = [obj.calc_reward(env) for obj in self._reward_obj_list]
        reward_tensor = torch.tensor(rewards, dtype=torch.float32)
        if self._cumulative_reward:
            self.episode_rewards.add_(reward_tensor)
        else:
            self.episode_rewards = reward_tensor
        self.episode_reward_history.append(rewards)

    def reset(self) -> None:
        """Reset for new episode."""
        self.episode_rewards = torch.zeros(len(self._reward_obj_list))
        self.episode_reward_history: list[list[float]] = []
        for obj in self._reward_obj_list:
            obj.reset()

        self._reward_weights = self._reward_initial_weights.clone()

    def set_multipliers(
            self, multipliers: np.ndarray | torch.Tensor | list, verbose: bool = False
    ) -> None:
        self._reward_calibration = torch.ones(len(self._reward_obj_list)) * multipliers
        if verbose:
            logger.info("New reward calibration weights: %s", self._reward_calibration)

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
        return (
            (self.episode_rewards * self._reward_weights * self._reward_calibration)
            .sum()
            .item()
        )

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
