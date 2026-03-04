"""Learning rate and noise schedulers for evolution strategies.

This module provides schedulers for controlling learning rate (alpha)
and noise standard deviation (sigma) during training.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class LearningScheduler:
    """Scheduler for evolution strategies hyperparameters.

    Implements exponential decay for learning rate (alpha) and noise
    standard deviation (sigma) with configurable floors.

    Attributes:
        alpha: Current learning rate.
        sigma: Current noise standard deviation.

    Example:
        >>> scheduler = LearningScheduler(initial_alpha=0.2, initial_sigma=0.1)
        >>> for epoch in range(100):
        ...     # Use scheduler.alpha and scheduler.sigma
        ...     scheduler.step()
    """

    # Initial values
    initial_alpha: float = 0.2
    initial_sigma: float = 0.2

    # Decay rates
    alpha_decay: float = 0.99
    sigma_decay: float = 0.99

    # Minimum values (floors)
    min_alpha: float = 0.0001
    min_sigma: float = 0.01

    # Current values (set in post_init)
    alpha: float = field(init=False)
    sigma: float = field(init=False)

    def __post_init__(self):
        """Initialize current values."""
        self.alpha = self.initial_alpha
        self.sigma = self.initial_sigma

    def step(self) -> None:
        """Apply one step of decay. Call at end of each epoch."""
        self.alpha = max(self.alpha * self.alpha_decay, self.min_alpha)
        self.sigma = max(self.sigma * self.sigma_decay, self.min_sigma)

    def reset(self) -> None:
        """Reset to initial values."""
        self.alpha = self.initial_alpha
        self.sigma = self.initial_sigma

    def state_dict(self) -> dict[str, Any]:
        """Get scheduler state for checkpointing.

        Returns:
            Dictionary with current state.
        """
        return {
            "alpha": self.alpha,
            "sigma": self.sigma,
            "initial_alpha": self.initial_alpha,
            "initial_sigma": self.initial_sigma,
            "alpha_decay": self.alpha_decay,
            "sigma_decay": self.sigma_decay,
            "min_alpha": self.min_alpha,
            "min_sigma": self.min_sigma,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Load scheduler state from checkpoint.

        Args:
            state: State dictionary from state_dict().
        """
        self.alpha = state["alpha"]
        self.sigma = state["sigma"]
        self.initial_alpha = state["initial_alpha"]
        self.initial_sigma = state["initial_sigma"]
        self.alpha_decay = state["alpha_decay"]
        self.sigma_decay = state["sigma_decay"]
        self.min_alpha = state["min_alpha"]
        self.min_sigma = state["min_sigma"]
