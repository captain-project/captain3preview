import logging
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from captain.agents.rewards import Rewards

logger = logging.getLogger(__name__)


class GlobalBudgetManager:
    """Manages tracking and safety truncation for global single-target runs."""

    def __init__(
        self,
        total_target: int,
        cells_per_time_step: int,
        feature_updates_per_time_step: int = 1,
    ):
        self.total_target = total_target
        self.cells_per_time_step = cells_per_time_step
        self.feature_updates_per_time_step = feature_updates_per_time_step

        self.cells_per_step = cells_per_time_step // feature_updates_per_time_step

    def get_step_context(self, env) -> dict:
        remaining = self.total_target - int(env.protected_cells_mask.sum().item())
        n_cells = min(self.cells_per_step, remaining)
        return {"n_cells": n_cells, "done": n_cells == 0}

    def get_info(self, env) -> dict:
        return {
            "total_target": self.total_target,
            "cells_per_time_step": self.cells_per_time_step,
            "cells_per_step": self.cells_per_step,
            "protected_cells": env.protected_cells_mask.sum(),
            "protected_fraction": env.protected_cells_mask.sum().item() / env.n_cells,
        }


class RegionalBudgetManager:
    """Manages stateful tracking and safe step targets across separate regions."""

    def __init__(
        self,
        masks: dict,
        total_targets: dict,
        cells_per_time_step: dict,
        feature_updates_per_time_step: int = 1,
    ):
        self.masks = masks
        self.total_targets = total_targets
        self.feature_updates_per_time_step = feature_updates_per_time_step

        self.per_step_targets = {
            r_id: max(1, val // feature_updates_per_time_step)
            for r_id, val in cells_per_time_step.items()
        }

        self.cells_per_time_step = cells_per_time_step

    def get_step_context(self, env) -> dict:
        step_k = {}
        done = True
        for r_id, total in self.total_targets.items():
            # Ensure mask is on the correct device/type if necessary
            mask = self.masks[r_id]
            if isinstance(mask, np.ndarray):
                mask = torch.from_numpy(mask)

            current = int(
                env.protected_cells_mask[mask.to(env.protected_cells_mask.device)]
                .sum()
                .item()
            )

            n_cells = min(self.per_step_targets[r_id], max(0, total - current))
            step_k[r_id] = n_cells
            if n_cells > 0:
                # if even one region still has budget for action it's not done
                done = False

        return {"region_masks": self.masks, "region_k": step_k, "done": done}

    def get_info(self, env) -> dict:
        return {
            "total_targets": self.total_targets,
            "cells_per_time_step": self.cells_per_time_step,
            "cells_per_step": self.per_step_targets,
            "protected_cells": env.protected_cells_mask.sum(),
            "protected_fraction": env.protected_cells_mask.sum().item() / env.n_cells,
        }


class NoBudgetManager:
    def __init__(self):
        self.total_target = 0
        self.cells_per_time_step = 0
        self.feature_updates_per_time_step = 1
        self.cells_per_step = 0

    def get_step_context(self, env) -> dict:
        return {"n_cells": 0, "done": True}

    def get_info(self, env) -> dict:
        return {
            "total_target": self.total_target,
            "cells_per_time_step": self.cells_per_time_step,
            "cells_per_step": self.cells_per_step,
            "protected_cells": env.protected_cells_mask.sum(),
            "protected_fraction": env.protected_cells_mask.sum().item() / env.n_cells,
        }
