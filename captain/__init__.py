"""CAPTAIN-NG: Evolutionary reinforcement learning for biodiversity conservation."""

from __future__ import annotations

__version__ = "3.0.0 beta - 20260821"

# Data structures
# Agents
from captain.agents.feature_extractor import FeatureExtractor
from captain.agents.feature_extractor_cnn import FeatureExtractorCNN
from captain.agents.policy_network import (
    CellCNNPolicy,
    CellNN,
    PolicyNetwork,
    RegionalPolicyNetwork,
)
from captain.agents.reward_aggregator import NoRewards, Rewards
from captain.agents.rewards import (
    CalcReward,
    CalcRewardDistFromTarget,
    CalcRewardExtRisk,
    CalcRewardPersistentCost,
    CalcRewardSpecieValue,
)
from captain.agents.spatial_cnn import SpatialCNN
from captain.algorithms import TrainingLogger
from captain.algorithms.budget_manager import (
    GlobalBudgetManager,
    NoBudgetManager,
    RegionalBudgetManager,
)

# Algorithms
from captain.algorithms.episode import BridgeEpisodeRunner, EpisodeRunner
from captain.algorithms.evolution_train import EvolStrategiesTrainer
from captain.algorithms.inference_recorder import InferenceRecorder
from captain.algorithms.scheduler import LearningScheduler
from captain.data.extinction_risk import ExtinctionRisk, ExtinctionRiskStatic
from captain.data.spatial_data import (
    SpatialData,
    StochasticSpatialData,
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
    "FeatureExtractorCNN",
    "CellNN",
    "CellCNNPolicy",
    "PolicyNetwork",
    "RegionalPolicyNetwork",
    "SpatialCNN",
    "CalcReward",
    "CalcRewardExtRisk",
    "CalcRewardDistFromTarget",
    "CalcRewardPersistentCost",
    "CalcRewardSpecieValue",
    "Rewards",
    "NoRewards",
    # Algorithms
    "EpisodeRunner",
    "BridgeEpisodeRunner",
    "InferenceRecorder",
    "EvolStrategiesTrainer",
    "LearningScheduler",
    "TrainingLogger",
    "GlobalBudgetManager",
    "RegionalBudgetManager",
    "NoBudgetManager",
    # Utilities
    "grid_utils",
    "plots",
    "data_loader",
]
