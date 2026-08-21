"""Episode runner for executing conservation simulations.

This module provides the EpisodeRunner class for running single episodes
of the conservation planning simulation.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from captain.utils.grid_utils import (
    compute_convolution_matrix,
    scipy_sparse_to_torch,
)

if TYPE_CHECKING:
    from captain.agents.feature_extractor import FeatureExtractor
    from captain.agents.feature_extractor_cnn import FeatureExtractorCNN
    from captain.agents.policy_network import PolicyNetwork
    from captain.agents.rewards import Rewards
    from captain.algorithms.budget_manager import (
        GlobalBudgetManager,
        NoBudgetManager,
        RegionalBudgetManager,
    )
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
            feature_extractor: FeatureExtractor | FeatureExtractorCNN,
            policy_network: PolicyNetwork,
            rewards: Rewards,
            budget_manager: GlobalBudgetManager | RegionalBudgetManager | NoBudgetManager,
            n_steps: int = 30,
            verbose: bool = False,
            save_protection_history: bool = False,
    ):
        """Initialize episode runner.

        Args:
            env: Biodiversity simulation environment.
            feature_extractor: Module for extracting features from env.
            policy_network: Network for selecting protection actions.
            rewards: Reward calculation module.
            budget_manager: Budget manager, checks how many cells can be protected
                            and apply global or regional budget (action ranking)
            n_steps: Total timesteps per episode.
            verbose: If True, print progress.
            save_protection_history: If True, record protection mask each step.
        """
        self.env = env
        self.feature_extractor = feature_extractor
        self.policy = policy_network
        self.rewards = rewards
        self.budget_manager = budget_manager
        self.n_steps = n_steps
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
            "extinction_risk": self.env.species_extinction_risk,
        }

    def run_episode(
            self,
            params: np.ndarray | torch.Tensor | None = None,
    ) -> tuple[dict[str, Any], float]:
        """Run a complete episode with optimized, conditional feature extraction."""
        # Reset state
        self.env.reset()
        if params is not None:
            self.policy.set_flat_weights(params)
        self.rewards.reset()
        self.protection_history = None

        # Flag to bypass the entire execution loop block once targets are hit
        protection_active = True

        with torch.no_grad():
            if self.save_protection_history:
                self.protection_history = self.env.protected_cells_mask.to(torch.int32)

            for t in range(self.n_steps):
                # Optimized Protection phase
                if protection_active:
                    for _ in range(self.budget_manager.feature_updates_per_time_step):
                        # Check A: What budgets does the manager allow right now?
                        budget_kwargs = self.budget_manager.get_step_context(self.env)

                        # Parse out if any allocation budget is left globally or regionally
                        has_budget = not (budget_kwargs["done"])

                        # Check B: Are there any open physical spaces left to act on?
                        # Using .all() on your boolean mask is an incredibly fast GPU tensor evaluation
                        cells_available = not self.env.no_action_mask.all()

                        # Short-circuit if either budget is met or the grid is filled
                        if not has_budget or not cells_available:
                            protection_active = False
                            break  # Breaks out of the micro-update loop

                        # --- EXPENSIVE COMPUTE LINE: Safe behind guards! ---
                        obs = self.feature_extractor.observe(self.env)

                        # Get actions polymorphically using Solution B signature
                        action = self.policy.get_actions(
                            obs,
                            constraint_mask=self.env.no_action_mask,
                            **budget_kwargs,
                        )

                        # Update manager tracking and apply to physics matrix
                        if len(action) > 0:
                            self.env.update_protection_matrix(action)

                        if self.save_protection_history:
                            self.protection_history += self.env.protected_cells_mask.to(
                                torch.int32
                            )

                # 4. Environment step (Always runs for the full length of n_steps)
                self.env.step()

                # 5. Calculate reward (currently only at each time step, not necessarily each action)
                self.rewards.calc_reward(self.env)

                if self.verbose:
                    logger.info(
                        f"Step: {t}/{self.n_steps} | "
                        f"Rewards: {self.rewards.episode_rewards} | "
                        f"Protected: {int(self.env.protected_cells_mask.sum().item())}"
                    )

        total_reward = self.rewards.get_weighted_reward()
        info = self.get_info()
        return info, total_reward


class BridgeEpisodeRunner(EpisodeRunner):
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
            feature_extractor: FeatureExtractor | FeatureExtractorCNN,
            policy_network: PolicyNetwork,
            rewards: Rewards,
            budget_manager: GlobalBudgetManager | RegionalBudgetManager | NoBudgetManager,
            n_steps: int = 30,
            verbose: bool = False,
            save_protection_history: bool = True,
    ):
        """Initialize episode runner.

        Args:
            env: Biodiversity simulation environment.
            feature_extractor: Module for extracting features from env.
            policy_network: Network for selecting protection actions.
            rewards: Reward calculation module.
            budget_manager: Budget manager, checks how many cells can be protected
                            and apply global or regional budget (action ranking)
            n_steps: Total timesteps per episode.
            verbose: If True, print progress.
            save_protection_history: If True, record protection mask each step.
        """
        super().__init__(
            env,
            feature_extractor,
            policy_network,
            rewards,
            budget_manager,
            n_steps,
            verbose,
            save_protection_history,
        )

        scipy_conv = compute_convolution_matrix(self.env.disturbance._coords, radius=1)
        self._conv_matrix = scipy_sparse_to_torch(scipy_conv, self.env.device)

        """Add edge reward only after connecting the bridge"""
        self._edge_reward_indx = None
        self._bridge_reward_indx = None
        for i in range(len(rewards._reward_obj_list)):
            if rewards._reward_obj_list[i]._name == "edge_effect":
                self._edge_reward_indx = i
            elif rewards._reward_obj_list[i]._name == "distance_from_target":
                self._bridge_reward_indx = i
            else:
                pass

    def run_episode(
            self,
            params: np.ndarray | torch.Tensor | None = None,
    ) -> tuple[dict[str, Any], float]:
        """Run a complete episode with optimized, conditional feature extraction."""
        # Reset state
        self.env.reset()
        if params is not None:
            self.policy.set_flat_weights(params)
        self.rewards.reset()
        self.protection_history = None

        # Flag to bypass the entire execution loop block once targets are hit
        protection_active = True

        with torch.no_grad():
            if self.save_protection_history:
                self.protection_history = self.env.protected_cells_mask.to(torch.int32)

            bridge_done = False
            dist_to_target = None
            for t in range(self.n_steps):
                # Optimized Protection phase
                if protection_active:
                    for _ in range(self.budget_manager.feature_updates_per_time_step):
                        # Check A: What budgets does the manager allow right now?
                        budget_kwargs = self.budget_manager.get_step_context(self.env)

                        # Parse out if any allocation budget is left globally or regionally
                        has_budget = not (budget_kwargs["done"])

                        # Check B: Are there any open physical spaces left to act on?
                        # Using .all() on your boolean mask is an incredibly fast GPU tensor evaluation
                        cells_available = not self.env.no_action_mask.all()

                        # Short-circuit if either budget is met or the grid is filled
                        if not has_budget or not cells_available:
                            protection_active = False
                            break  # Breaks out of the micro-update loop

                        # --- update features ---
                        if not bridge_done:
                            # calculate distances to target among actionable cells
                            dist_to_target = self.env.get_distance_from_target(
                                normalize=True
                            )
                            if dist_to_target is not None:
                                self.feature_extractor.reset_static_features(
                                    dist_to_target, indices=0
                                )
                        else:
                            if dist_to_target is not None:
                                self.feature_extractor.reset_static_features(
                                    torch.zeros(dist_to_target.shape), indices=0
                                )

                        obs = self.feature_extractor.observe(self.env)

                        # Get actions polymorphically using Solution B signature
                        action = self.policy.get_actions(
                            obs,
                            constraint_mask=self.env.no_action_mask,
                            **budget_kwargs,
                        )

                        # Update manager tracking and apply to physics matrix
                        if len(action) > 0:
                            self.env.update_protection_matrix(action)

                        if self.save_protection_history:
                            self.protection_history += self.env.protected_cells_mask.to(
                                torch.int32
                            )

                        # update the no action mask based on current protection matrix
                        # (excluding existing protected areas)
                        newly_protected_cells = (
                                                        self.protection_history < self.protection_history.max()
                                                ).to(torch.int32) * (self.protection_history > 0).to(
                            torch.int32
                        )

                        # Sparse matrix multiplication: (n_cells,) @ sparse(n_cells, n_cells)
                        # Note: For MPS, sparse matrix is on CPU
                        sparse_device = self._conv_matrix.device
                        if sparse_device != self.env.device:
                            # MPS path: transfer to CPU for sparse op, then back
                            newly_protected_cells_cpu = newly_protected_cells.to(
                                sparse_device
                            )
                            conv_result = torch.sparse.mm(
                                self._conv_matrix,
                                newly_protected_cells_cpu.unsqueeze(1).float(),
                            ).squeeze(1)
                        else:
                            conv_result = torch.sparse.mm(
                                self._conv_matrix,
                                newly_protected_cells.unsqueeze(1).float(),
                            ).squeeze(1)

                        # adjacent_mask = 1: no action possible, adjacent_mask = 0: space of action
                        adjacent_mask = ~(
                                (conv_result > 0) & (self.env.protection_matrix.data == 0)
                        )

                        self.env.action_mask.update_cell_values(
                            np.arange(self.env.n_cells), adjacent_mask.to(torch.float32)
                        )

                        self.rewards.calc_reward(self.env)
                        if self._bridge_reward_indx is not None:
                            # check if bridge was built
                            if self.rewards._reward_obj_list[
                                self._bridge_reward_indx
                            ]._target_achieved:
                                # set edge reward weight to 1.0
                                if not bridge_done:
                                    print(
                                        f"Target achieved with {self.env.protection_matrix.data.sum()} cells ({self.rewards._reward_weights})",
                                        self.rewards.get_weighted_reward(),
                                    )
                                self.rewards._reward_weights[self._edge_reward_indx] = (
                                    1.0
                                )
                                if not bridge_done:
                                    print(
                                        f"{self.rewards._reward_weights}",
                                        self.rewards.get_weighted_reward(),
                                    )
                                bridge_done = True
                            else:
                                # print("Target NOT achieved", self.rewards._reward_weights)
                                self.rewards._reward_weights[self._edge_reward_indx] = (
                                    0.0
                                )
                                # self.rewards.calc_reward(self.env)

                # 4. Environment step (Always runs for the full length of n_steps)
                self.env.step()

                # 5. Calculate reward
                # self.rewards.calc_reward(self.env)

                if self.verbose:
                    logger.info(
                        f"Step: {t}/{self.n_steps} | "
                        f"Rewards: {self.rewards.episode_rewards} | "
                        f"Protected: {int(self.env.protected_cells_mask.sum().item())}"
                    )

        total_reward = self.rewards.get_weighted_reward()
        info = self.get_info()
        return info, total_reward
