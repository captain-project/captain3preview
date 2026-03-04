"""Episode runner for executing conservation simulations.

This module provides the EpisodeRunner class for running single episodes
of the conservation planning simulation.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

if TYPE_CHECKING:
    from captain.agents.feature_extractor import FeatureExtractor
    from captain.agents.policy_network import PolicyNetwork
    from captain.agents.rewards import Rewards
    from captain.environment.bioenv import BioEnv

logger = logging.getLogger(__name__)


class EpisodeRunner:
    """Runs a single episode of the conservation simulation.

    Orchestrates the interaction between environment, policy network, and
    reward calculation for one complete episode.

    Attributes:
        env: Biodiversity environment.
        feature_extractor: Feature extraction module.
        policy: Policy network for action selection.
        rewards: Reward calculation module.

    Example:
        >>> runner = EpisodeRunner(env, extractor, policy, rewards)
        >>> info, total_reward = runner.run_episode(weights)
    """

    def __init__(
            self,
            env: BioEnv,
            feature_extractor: FeatureExtractor,
            policy_network: PolicyNetwork,
            rewards: Rewards,
            n_steps: int = 30,
            n_total_protected_cells: int = 100,
            n_protected_cells_per_time_step: int = 10,
            feature_updates_per_time_step: int = 1,
            verbose: bool = False,
            save_protection_history: bool = False,
    ):
        """Initialize episode runner.

        Args:
            env: Biodiversity simulation environment.
            feature_extractor: Module for extracting features from env.
            policy_network: Network for selecting protection actions.
            rewards: Reward calculation module.
            n_steps: Total timesteps per episode.
            n_total_protected_cells: Target number of protected cells.
            n_protected_cells_per_time_step: Total cells to protect per timestep.
            feature_updates_per_time_step: How many times to recompute features per timestep.
            verbose: If True, print progress.
        """
        self.env = env
        self.feature_extractor = feature_extractor
        self.policy = policy_network
        self.rewards = rewards
        self.n_steps = n_steps
        self.n_total_protected_cells = n_total_protected_cells
        self.n_protected_cells_per_step = (
                n_protected_cells_per_time_step // feature_updates_per_time_step
        )
        self.n_protected_cells_per_time_step = n_protected_cells_per_time_step
        self.feature_updates_per_time_step = feature_updates_per_time_step
        self.verbose = verbose
        self.save_protection_history = save_protection_history
        self.protection_history = None

    def get_info(self) -> dict[str, Any]:
        """Get episode information.

        Returns:
            Dictionary with episode statistics.
        """
        return {
            "rewards": (
                self.rewards.episode_rewards.clone()
                if isinstance(self.rewards.episode_rewards, torch.Tensor)
                else self.rewards.episode_rewards.copy()
            ),
            "reward_history": list(self.rewards.episode_reward_history),
            "n_steps": self.n_steps,
            "protected_cells": int(self.env.protected_cells_mask.sum().item()),
            "protection_matrix": self.env.protection_matrix.reconstruct_grid.copy(),
            "protection_history": self.protection_history,
        }

    def run_episode(
            self,
            params: np.ndarray | torch.Tensor | None = None,
    ) -> tuple[dict[str, Any], float]:
        """Run a complete episode.

        Args:
            params: Policy weights to set before running.

        Returns:
            Tuple of (info dict, total reward).
        """
        # Reset state
        self.env.reset()
        if params is not None:
            self.policy.set_flat_weights(params)
        self.rewards.reset()
        self.protection_history = None

        with torch.no_grad():

            if self.save_protection_history:
                self.protection_history = self.env.protected_cells_mask.to(torch.int32)
            for t in range(self.n_steps):
                current_protected_cells = int(
                    self.env.protected_cells_mask.sum().item()
                )

                # Protection phase
                if current_protected_cells < self.n_total_protected_cells:
                    for _ in range(self.feature_updates_per_time_step):
                        # 1. Observe
                        obs = self.feature_extractor.observe(self.env)

                        # 2. Get actions
                        action = self.policy.get_actions(
                            obs,
                            self.n_protected_cells_per_step,
                            constraint_mask=self.env.protected_cells_mask,
                        )

                        current_protected_cells += self.n_protected_cells_per_step
                        d = current_protected_cells - self.n_total_protected_cells

                        if d > 0:
                            # Don't exceed total if not divisible
                            action = action[:-d]

                        # 3. Apply protection
                        self.env.update_protection_matrix(action)

                        if self.save_protection_history:
                            self.protection_history += self.env.protected_cells_mask.to(
                                torch.int32
                            )

                # 4. Environment step
                self.env.step()

                # 5. Calculate reward
                self.rewards.calc_reward(self.env)

                if self.verbose:
                    logger.info(
                        f"Step: {t}/{self.n_steps} | "
                        f"Rewards: {self.rewards.episode_rewards} | "
                        f"Protected: {current_protected_cells}"
                    )

        total_reward = self.rewards.get_weighted_reward()
        info = self.get_info()
        return info, total_reward
