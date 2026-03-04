"""Policy network for cell selection in conservation planning.

This module provides the neural network architecture and wrapper class for
scoring cells and selecting optimal locations for protection.
"""

from __future__ import annotations

import logging

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class CellNN(nn.Module):
    """Neural network for scoring individual cells.

    A simple MLP that takes cell features and outputs a scalar score
    indicating the priority for protection.

    Architecture:
        Input: (n_features, n_cells)
        -> Transpose to (n_cells, n_features)
        -> Linear(n_features, hidden_dim)
        -> ReLU
        -> Linear(hidden_dim, 1)
        -> Squeeze to (n_cells,)

    Example:
        >>> model = CellNN(input_dim=13, hidden_dim=32)
        >>> features = torch.randn(13, 1000)  # 13 features, 1000 cells
        >>> scores = model(features)  # shape: (1000,)
    """

    def __init__(
            self,
            input_dim: int,
            hidden_dim: int = 32,
            activation: str = "relu",
    ):
        """Initialize cell scoring network.

        Args:
            input_dim: Number of input features per cell.
            hidden_dim: Hidden layer dimension.
            activation: Activation function ('relu', 'tanh', 'gelu').
        """
        super().__init__()

        # Select activation
        activations = {
            "relu": nn.ReLU(),
            "tanh": nn.Tanh(),
            "gelu": nn.GELU(),
        }
        if activation not in activations:
            raise ValueError(
                f"Unknown activation: {activation}. Use one of {list(activations)}"
            )

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            activations[activation],
            nn.Linear(hidden_dim, 1),
        )

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Features tensor of shape (n_features, n_cells).

        Returns:
            Scores tensor of shape (n_cells,).
        """
        # Transpose: (f, n) -> (n, f)
        x = x.t()
        # Forward: (n, f) -> (n, 1)
        out = self.net(x)
        # Squeeze: (n, 1) -> (n,)
        return out.squeeze(-1)


class PolicyNetwork:
    """Wrapper for cell-scoring neural network with weight management.

    Provides methods for weight extraction/injection (for evolution strategies)
    and action selection with constraint handling.

    Attributes:
        model: The underlying CellNN network.
        device: PyTorch device for computations.

    Example:
        >>> model = CellNN(input_dim=13, hidden_dim=32)
        >>> policy = PolicyNetwork(model, device="cuda")
        >>> actions = policy.get_actions(features, n_cells=10, constraint_mask=mask)
    """

    # Tie-breaking noise scale
    TIE_BREAK_NOISE: float = 1e-7

    def __init__(
            self,
            model: nn.Module,
            device: torch.device | str = "cpu",
            seed: int | None = None,
    ):
        """Initialize policy network wrapper.

        Args:
            model: CellNN model or compatible module.
            device: PyTorch device.
        """
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.gen = torch.Generator(device=self.device)
        if seed is not None:
            self.gen.manual_seed(seed)
            self.seeded_init(model, seed)

    def seeded_init(self, model, seed):
        # Save the current random state
        current_state = torch.random.get_rng_state()

        # Set the seed for the init phase
        torch.manual_seed(seed)

        # Apply standard torch init methods
        for m in model.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        # Restore the previous random state
        torch.random.set_rng_state(current_state)

    def get_flat_weights(self) -> np.ndarray:
        """Extract all model parameters as a flat 1D NumPy array.

        Returns:
            1D array of all model parameters concatenated.
        """
        with torch.no_grad():
            parameters = nn.utils.parameters_to_vector(self.model.parameters())
            return parameters.cpu().numpy()

    def set_flat_weights(self, new_weights: np.ndarray | torch.Tensor) -> None:
        """Inject a flat weight vector into model parameters.

        Args:
            new_weights: 1D array of weights to inject.
        """
        if isinstance(new_weights, np.ndarray):
            new_weights = torch.from_numpy(new_weights).float()

        new_weights = new_weights.to(self.device)

        with torch.no_grad():
            nn.utils.vector_to_parameters(new_weights, self.model.parameters())

    def get_n_params(self) -> int:
        """Get total number of trainable parameters.

        Returns:
            Number of parameters.
        """
        return sum(p.numel() for p in self.model.parameters())

    def select_k_cells(self, scores: torch.Tensor, n_cells: int) -> torch.Tensor:
        """Select top-k cells by score with tie-breaking noise.

        Args:
            scores: Cell scores tensor of shape (n_cells,).
            n_cells: Number of cells to select.

        Returns:
            Indices of selected cells.
        """
        # Add tiny noise to break ties
        noise = torch.randn_like(scores, generator=self.gen) * self.TIE_BREAK_NOISE
        _, top_indices = torch.topk(scores + noise, n_cells, sorted=False)
        return top_indices

    def get_scores(self, observation: torch.Tensor | np.ndarray) -> torch.Tensor:
        """Compute cell scores from observations.

        Args:
            observation: Feature tensor of shape (n_features, n_cells).

        Returns:
            Score tensor of shape (n_cells,).
        """
        if isinstance(observation, np.ndarray):
            observation = torch.from_numpy(observation).float()

        observation = observation.to(self.device)

        with torch.no_grad():
            scores = self.model(observation)

        return scores

    def get_actions(
            self,
            observation: torch.Tensor | np.ndarray,
            n_cells: int,
            constraint_mask: torch.Tensor | np.ndarray | None = None,
    ) -> torch.Tensor:
        """Get indices of cells to protect.

        Args:
            observation: Feature tensor of shape (n_features, n_cells).
            n_cells: Number of cells to select.
            constraint_mask: Boolean mask of cells to exclude (True = exclude).

        Returns:
            Indices of selected cells.
        """
        scores = self.get_scores(observation)

        if constraint_mask is not None:
            if isinstance(constraint_mask, np.ndarray):
                constraint_mask = torch.from_numpy(constraint_mask)
            constraint_mask = constraint_mask.to(self.device)
            scores = scores.masked_fill(constraint_mask, float("-inf"))

        return self.select_k_cells(scores, n_cells)

    def to(self, device: torch.device | str) -> PolicyNetwork:
        """Move policy to specified device.

        Args:
            device: Target device.

        Returns:
            Self for chaining.
        """
        self.device = torch.device(device)
        self.model = self.model.to(self.device)
        self.gen = torch.Generator(device=self.device)
        return self

    def save(self, path: str) -> None:
        """Save model weights to file.

        Args:
            path: Path to save weights.
        """
        torch.save(self.model.state_dict(), path)
        logger.info(f"Saved policy weights to {path}")

    def load(self, path: str) -> None:
        """Load model weights from file.

        Args:
            path: Path to load weights from.
        """
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        logger.info(f"Loaded policy weights from {path}")
