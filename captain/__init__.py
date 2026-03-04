"""CAPTAIN-NG: Evolutionary reinforcement learning for biodiversity conservation."""

from __future__ import annotations

__version__ = "3.0.0 beta - 20260304"

# Data structures
# Agents
from captain.agents.feature_extractor import FeatureExtractor
from captain.agents.policy_network import CellNN, PolicyNetwork
from captain.agents.rewards import (
    CalcReward,
    CalcRewardExtRisk,
    CalcRewardPersistentCost,
    Rewards,
)

# Algorithms
from captain.algorithms.episode import EpisodeRunner
from captain.algorithms.evolution_train import EvolStrategiesTrainer
from captain.algorithms.scheduler import LearningScheduler
from captain.data.extinction_risk import ExtinctionRisk, ExtinctionRiskStatic
from captain.data.spatial_data import (
    SpatialData,
    load_spatial_data,
    load_spatial_data_from_dir,
)

# Environment
from captain.environment.bioenv import BioEnv

# Utilities
from captain.utils import data_loader, grid_utils, plots

__all__ = [
    # Version
    "__version__",
    # Data
    "SpatialData",
    "ExtinctionRisk",
    "ExtinctionRiskStatic",
    "load_spatial_data",
    "load_spatial_data_from_dir",
    # Environment
    "BioEnv",
    # Agents
    "FeatureExtractor",
    "CellNN",
    "PolicyNetwork",
    "CalcReward",
    "CalcRewardExtRisk",
    "CalcRewardPersistentCost",
    "Rewards",
    # Algorithms
    "EpisodeRunner",
    "EvolStrategiesTrainer",
    "LearningScheduler",
    # Utilities
    "grid_utils",
    "plots",
    "data_loader",
]
