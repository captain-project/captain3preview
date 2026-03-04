"""Evolution strategies trainer for policy optimization.

This module implements Natural Evolution Strategies (NES) for training
conservation policies without gradients.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import os
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from captain.algorithms import scheduler as sched

if TYPE_CHECKING:
    from captain.algorithms.episode import EpisodeRunner

logger = logging.getLogger(__name__)


def compute_evolutionary_update(
        results: list[tuple[dict, float]],
        epoch_coeff: np.ndarray,
        noise: np.ndarray,
        alpha: float,
        sigma: float,
        running_reward: float,
) -> np.ndarray:
    """Compute evolution strategies weight update.

    Args:
        results: List of (info, reward) tuples from parallel evaluation.
        epoch_coeff: Current weight vector.
        noise: Noise perturbations used for this epoch.
        alpha: Learning rate.
        sigma: Noise standard deviation.
        running_reward: Baseline reward for advantage computation.

    Returns:
        Updated weight vector.
    """
    if sigma == 0:
        return epoch_coeff

    # Extract rewards (handle both scalar and array rewards)
    rewards = np.array([np.sum(r[1]) for r in results])
    n = len(rewards)

    # Compute advantage-weighted noise
    perturbed_advantage = [
        (rr - running_reward) * nn for rr, nn in zip(rewards, noise, strict=True)
    ]

    # Update weights
    new_coeff = epoch_coeff + (alpha / (n * sigma)) * np.sum(perturbed_advantage, 0)

    return new_coeff


# Global worker state
_runner: EpisodeRunner | None = None


def setup_worker(runner: EpisodeRunner) -> None:
    """Initialize worker with episode runner.

    Called once per worker process to set up persistent state.

    Args:
        runner: EpisodeRunner instance for this worker.
    """
    global _runner
    torch.set_num_threads(1)  # Prevent thread oversubscription
    _runner = runner


def execute_task(params: np.ndarray) -> tuple[dict[str, Any], float]:
    """Execute one episode with given parameters.

    Args:
        params: Policy weights to evaluate.

    Returns:
        Tuple of (info dict, total reward).
    """
    global _runner
    if _runner is None:
        raise RuntimeError("Worker not initialized. Call setup_worker first.")
    return _runner.run_episode(params)


def summarize_episodes(results: list[tuple[dict, float]]) -> dict[str, Any]:
    """Summarize info from all parallel episodes.

    Args:
        results: List of (info, total_reward) tuples.

    Returns:
        Dictionary with averaged metrics.
    """
    infos = [res[0] for res in results]

    # Average rewards by type
    all_rewards = np.array(
        [
            (
                i["rewards"].numpy()
                if isinstance(i["rewards"], torch.Tensor)
                else np.array(i["rewards"])
            )
            for i in infos
        ]
    )
    avg_rewards = np.mean(all_rewards, axis=0)

    # Average reward history
    all_histories = np.array([i["reward_history"] for i in infos])
    avg_history = np.mean(all_histories, axis=0)

    # Average protected cells
    avg_protected = np.mean([i["protected_cells"] for i in infos])

    avg_protection_matrix = np.mean(
        np.array([i["protection_matrix"] for i in infos]), axis=0
    )

    return {
        "avg_rewards_by_type": avg_rewards,
        "avg_reward_history": avg_history,
        "avg_protected_cells": avg_protected,
        "avg_protection_matrix": avg_protection_matrix,
    }


class EvolStrategiesTrainer:
    """Natural Evolution Strategies trainer for policy optimization.

    Uses parallel episode evaluation and fitness-weighted averaging to
    optimize policy weights without computing gradients.

    Attributes:
        epoch_coeff: Current policy weight vector.
        running_reward: Exponential moving average of rewards.
        scheduler: Learning rate and noise scheduler.

    Example:
        >>> trainer = EvolStrategiesTrainer(runners, initial_weights)
        >>> for epoch in range(100):
        ...     reward, summary = trainer.train_epoch()
        ...     print(f"Epoch {epoch}: {reward:.4f}")
    """

    def __init__(
            self,
            list_of_env_params: list[EpisodeRunner],
            initial_coeffs: np.ndarray,
            scheduler: sched.LearningScheduler | None = None,
            epsilon_reward: float = 0.5,
            n_perturbations: int | None = None,
            seed: int | None = None,
    ):
        """Initialize trainer.

        Args:
            list_of_env_params: List of EpisodeRunner instances, one per worker.
                Pass a single-element list to use sequential mode (no pool).
            initial_coeffs: Initial policy weight vector.
            scheduler: Learning rate/noise scheduler (default: create new).
            epsilon_reward: EMA smoothing factor for baseline.
            n_perturbations: Number of ES noise samples per epoch. Defaults to
                ``len(list_of_env_params)`` for backward compatibility. When using
                sequential mode (single runner), this should be set explicitly.
            seed: Random seed for reproducibility.
        """
        self.epoch_coeff = initial_coeffs.astype(np.float32).copy()
        self.scheduler = scheduler or sched.LearningScheduler()
        self.running_reward: float | None = None
        self.epsilon_reward = epsilon_reward
        self.rg = np.random.default_rng(seed)

        if len(list_of_env_params) == 1:
            # Sequential mode — run episodes in a loop on the current process.
            # This avoids pickling runners, which is required for GPU tensors.
            self._use_pool = False
            self._runner = list_of_env_params[0]
            self.n = n_perturbations if n_perturbations is not None else 1
            logger.info(
                f"Initialized sequential trainer with {self.n} perturbations, "
                f"{len(self.epoch_coeff)} parameters"
            )
        else:
            # Pool mode — distribute runners across worker processes.
            self._use_pool = True
            self.n = (
                n_perturbations
                if n_perturbations is not None
                else len(list_of_env_params)
            )
            n_workers = min(len(list_of_env_params), mp.cpu_count())

            # Use forkserver to prevent OpenMP deadlock.
            # With the default "fork" start method, child processes inherit
            # the parent's OpenMP/MKL thread-pool state. Since PyTorch
            # typically initialises OpenMP with many threads before the pool
            # is created, the forked children deadlock on the first tensor
            # operation.  "forkserver" avoids this by spawning workers from
            # a clean server process that has not yet initialised OMP.
            if os.name == "posix":  # For Unix-based systems (macOS, Linux)
                ctx = mp.get_context("forkserver")
            else:  # For Windows
                ctx = mp.get_context("spawn")

            self.pool = ctx.Pool(processes=n_workers)
            self.pool.map(setup_worker, list_of_env_params)
            logger.info(
                f"Initialized pool trainer with {n_workers} workers, "
                f"{self.n} perturbations, {len(self.epoch_coeff)} parameters"
            )

    def train_epoch(self) -> tuple[float, dict[str, Any]]:
        """Run one training epoch.

        Returns:
            Tuple of (average reward, episode summary).
        """
        n_params = len(self.epoch_coeff)

        # 1. Generate noise perturbations
        noise = self.rg.standard_normal((self.n, n_params)).astype(np.float32)

        # 2. Create perturbed parameter vectors
        params = [
            self.epoch_coeff + self.scheduler.sigma * noise[i] for i in range(self.n)
        ]

        # 3. Evaluate perturbations
        if self._use_pool:
            results = self.pool.map(execute_task, params)
        else:
            results = [self._runner.run_episode(p) for p in params]

        # 4. Initialize baseline if first epoch
        if self.running_reward is None:
            self.running_reward = float(np.mean([np.sum(r[1]) for r in results]))

        # 5. Compute update
        new_coeff = compute_evolutionary_update(
            results=results,
            epoch_coeff=self.epoch_coeff,
            noise=noise,
            alpha=self.scheduler.alpha,
            sigma=self.scheduler.sigma,
            running_reward=self.running_reward,
        )
        self.epoch_coeff = new_coeff

        # 6. Update baseline (EMA)
        avg_reward = float(np.mean([np.sum(r[1]) for r in results]))
        self.running_reward = (
                self.epsilon_reward * avg_reward
                + (1 - self.epsilon_reward) * self.running_reward
        )

        # 7. Update scheduler
        self.scheduler.step()

        # 8. Summarize
        summary = summarize_episodes(results)

        return avg_reward, summary

    def get_weights(self) -> np.ndarray:
        """Get current policy weights.

        Returns:
            Current weight vector.
        """
        return self.epoch_coeff.copy()

    def close(self) -> None:
        """Clean up worker pool (no-op in sequential mode)."""
        if self._use_pool and hasattr(self, "pool"):
            self.pool.close()
            self.pool.join()

    def __del__(self):
        """Ensure pool is closed on deletion."""
        self.close()
