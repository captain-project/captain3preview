"""Recording of policy inputs and cell rankings during inference.

This module provides the InferenceRecorder class, which can be attached to an
EpisodeRunner to save, at every policy decision, the features the policy saw,
the scores it produced, and the resulting cell ranking. The output is meant for
downstream explainability (xAI) analyses and is not used during training.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import torch

if TYPE_CHECKING:
    from captain.agents.feature_extractor import FeatureExtractor
    from captain.agents.feature_extractor_cnn import FeatureExtractorCNN
    from captain.agents.policy_network import PolicyNetwork
    from captain.environment.bioenv import BioEnv

logger = logging.getLogger(__name__)


def _to_numpy(x: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def rank_cells(
    scores: np.ndarray,
    eligible: np.ndarray,
    groups: Sequence[np.ndarray] | None = None,
) -> np.ndarray:
    """Rank eligible cells by descending score.

    Args:
        scores: Cell scores of shape (n_cells,).
        eligible: Boolean mask of shape (n_cells,), True where a cell can be selected.
        groups: Optional boolean masks of shape (n_cells,). If given, cells are
            ranked independently within each group (e.g. regional agents), and
            eligible cells outside every group get rank -1.

    Returns:
        Integer array of shape (n_cells,) where 1 is the highest priority and
        -1 marks cells that are not ranked.
    """
    rank = np.full(scores.shape[0], -1, dtype=np.int32)
    if groups is None:
        groups = [np.ones_like(eligible, dtype=bool)]
    for group in groups:
        idx = np.flatnonzero(eligible & group)
        if idx.size == 0:
            continue
        order = idx[np.argsort(-scores[idx], kind="stable")]
        rank[order] = np.arange(1, idx.size + 1, dtype=np.int32)
    return rank


class InferenceRecorder:
    """Saves features, scores and cell rankings at each policy decision.

    Pass an instance to ``EpisodeRunner(recorder=...)``. For each recorded
    decision a file ``step_{t:03d}_u{u:02d}.npz`` is written to ``out_dir``
    (``t`` = time step, ``u`` = feature update within the time step), containing:

    - ``features_input`` (F, N) or (F, M): features as fed to the network (normalized).
    - ``features_raw`` (F, N) or (F, M): features before normalization (only for
      extractors exposing a raw ``features`` buffer, i.e. ``FeatureExtractor``).
    - ``feature_names`` (F,): feature names.
    - ``sample_cell_idx`` (M,), ``sample_is_top`` (M,), ``sample_inclusion_prob`` (M,):
      only when ``feature_sample_fraction`` is set; see below.
    - ``scores`` (N,): raw network output, before masking.
    - ``eligible`` (N,): cells that could be selected at this decision.
    - ``already_protected`` (N,): cells protected before this decision.
    - ``selected`` (N,): cells selected at this decision.
    - ``rank`` (N,): 1 = highest priority among eligible cells, -1 = not eligible.
      With regional agents, cells are ranked within their region.
    - ``region_id`` (N,): index into ``region_names`` (-1 = no region); regional
      agents only.
    - ``time_step``, ``update_idx``: decision indices.

    The paths of the step files written in the last episode are listed in ``files``
    (e.g. to plot them with ``plots.plot_feature_scores_from_file``).

    A single ``cells.npz`` holds the grid coordinates of each cell
    (``coords_row``, ``coords_col``), ``grid_shape``, ``feature_names`` and, for
    regional agents, ``region_names``.

    Feature subsampling: with ``feature_sample_fraction`` set, per-cell arrays
    (scores, masks, rank) are still saved for all N cells, but features are saved
    only for M sampled cells. Among the eligible cells of each ranking scope (the
    whole grid, or each region for regional agents), with ``n`` = max(ceil(fraction *
    n_eligible), n_selected), the ``n`` top-ranked cells are saved together with
    ``n`` cells drawn at random (fresh at each decision) from the remaining eligible
    cells. ``sample_inclusion_prob`` is 1 for top cells and n / n_remaining for
    random cells; use ``1 / sample_inclusion_prob`` to reweight population statistics.

    Once the protection budget is exhausted the policy is no longer queried, so
    later time steps produce no files.

    Example:
        >>> recorder = InferenceRecorder("results/xai", steps=[0, 10])
        >>> runner = EpisodeRunner(..., recorder=recorder)
        >>> runner.run_episode(weights)
        >>> d = np.load("results/xai/step_000_u00.npz")
        >>> cells = np.load("results/xai/cells.npz")
        >>> rank_map = grid_utils.reconstruct_grid(
        ...     d["rank"][None], (cells["coords_row"], cells["coords_col"]),
        ...     tuple(cells["grid_shape"]))
    """

    CELLS_FILE = "cells.npz"

    def __init__(
        self,
        out_dir: str | Path,
        steps: Literal["all"] | int | Sequence[int] = "all",
        save_raw_features: bool = True,
        compress: bool = False,
        feature_sample_fraction: float | None = None,
        seed: int | None = None,
    ):
        """Initialize recorder.

        Args:
            out_dir: Directory where npz files are written (created if missing).
            steps: Time steps to record: ``"all"``, a single step, or a sequence of steps.
            save_raw_features: If True, also save un-normalized features when available.
            compress: If True, use ``np.savez_compressed`` (smaller, slower).
            feature_sample_fraction: If None, save features for all cells. Otherwise,
                fraction of eligible cells defining the size of the top-ranked and
                random samples (see class docstring).
            seed: Seed for the random sample of cells (reset at each episode).

        Raises:
            ValueError: If ``steps`` or ``feature_sample_fraction`` are invalid.
        """
        if feature_sample_fraction is not None and not 0 < feature_sample_fraction <= 1:
            raise ValueError(
                f"feature_sample_fraction must be in (0, 1], got {feature_sample_fraction}"
            )
        self.feature_sample_fraction = feature_sample_fraction
        self.seed = seed
        self.out_dir = Path(out_dir)
        if steps == "all":
            self.steps: set[int] | None = None
        elif isinstance(steps, (int, np.integer)):
            self.steps = {int(steps)}
        elif isinstance(steps, Sequence) and not isinstance(steps, str):
            self.steps = {int(s) for s in steps}
        else:
            raise ValueError(f"steps must be 'all', an int or a sequence of ints, got {steps!r}")
        self.save_raw_features = save_raw_features
        self.compress = compress
        self.reset()

    def reset(self) -> None:
        """Clear per-episode bookkeeping (called at the start of each episode)."""
        self._recorded_steps: set[int] = set()
        self._n_files = 0
        self._cells_written = False
        self._rng = np.random.default_rng(self.seed)
        self.files: list[Path] = []

    def should_record(self, t: int) -> bool:
        """Whether decisions at time step ``t`` should be recorded."""
        return self.steps is None or t in self.steps

    def _save(self, path: Path, **arrays: Any) -> None:
        if self.compress:
            np.savez_compressed(path, **arrays)
        else:
            np.savez(path, **arrays)

    def _sample_cells(
        self,
        rank: np.ndarray,
        eligible: np.ndarray,
        selected: np.ndarray,
        scopes: Sequence[np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Draw top-ranked and random eligible cells within each ranking scope.

        Returns:
            Tuple of (cell indices, is-top flags, inclusion probabilities).
        """
        assert self.feature_sample_fraction is not None
        idx, is_top, prob = [], [], []
        for scope in scopes:
            scope_idx = np.flatnonzero(eligible & scope)
            if scope_idx.size == 0:
                continue
            n = max(
                math.ceil(self.feature_sample_fraction * scope_idx.size),
                int((selected & scope).sum()),
            )
            n = min(n, scope_idx.size)
            ordered = scope_idx[np.argsort(rank[scope_idx], kind="stable")]
            top, rest = ordered[:n], ordered[n:]
            n_random = min(n, rest.size)
            random = np.sort(self._rng.choice(rest, size=n_random, replace=False))

            idx += [top, random]
            is_top += [np.ones(top.size, dtype=bool), np.zeros(n_random, dtype=bool)]
            prob += [
                np.ones(top.size, dtype=np.float32),
                np.full(n_random, n_random / max(rest.size, 1), dtype=np.float32),
            ]
        if not idx:
            return np.empty(0, np.int64), np.empty(0, bool), np.empty(0, np.float32)
        return np.concatenate(idx), np.concatenate(is_top), np.concatenate(prob)

    def record(
        self,
        t: int,
        update_idx: int,
        env: BioEnv,
        feature_extractor: FeatureExtractor | FeatureExtractorCNN,
        policy: PolicyNetwork,
        obs: torch.Tensor,
        action: torch.Tensor,
        budget_kwargs: dict[str, Any],
    ) -> None:
        """Record one policy decision.

        Must be called after ``policy.get_actions`` and before the selected cells
        are applied to the environment, so that masks reflect what the policy saw.

        Args:
            t: Time step index.
            update_idx: Feature update index within the time step.
            env: Environment (state before applying ``action``).
            feature_extractor: Extractor that produced ``obs``.
            policy: Policy network that produced ``action``.
            obs: Observation passed to the policy, shape (n_features, n_cells).
            action: Indices of selected cells.
            budget_kwargs: Budget context passed to ``get_actions``.
        """
        self.out_dir.mkdir(parents=True, exist_ok=True)
        n_cells = env.n_cells

        features_input = _to_numpy(obs).astype(np.float32, copy=False)
        n_features = features_input.shape[0]
        names = getattr(feature_extractor, "feature_names", None)
        if names is None or len(names) != n_features:
            names = [f"feature_{k}" for k in range(n_features)]
        feature_names = np.array(names, dtype=str)

        scores = _to_numpy(policy.get_scores(obs)).astype(np.float32, copy=False)
        eligible = ~_to_numpy(env.no_action_mask).astype(bool)
        already_protected = _to_numpy(env.protected_cells_mask).astype(bool)
        selected = np.zeros(n_cells, dtype=bool)
        selected[_to_numpy(action).astype(np.int64)] = True

        arrays: dict[str, Any] = {
            "features_input": features_input,
            "feature_names": feature_names,
            "scores": scores,
            "eligible": eligible,
            "already_protected": already_protected,
            "selected": selected,
            "time_step": np.int32(t),
            "update_idx": np.int32(update_idx),
        }

        raw = getattr(feature_extractor, "features", None)
        if self.save_raw_features and isinstance(raw, torch.Tensor) and raw.shape == obs.shape:
            arrays["features_raw"] = _to_numpy(raw).astype(np.float32, copy=True)

        region_masks = budget_kwargs.get("region_masks")
        region_names = None
        if region_masks is not None:
            region_names = np.array([str(k) for k in region_masks], dtype=str)
            scopes = [_to_numpy(m).astype(bool) for m in region_masks.values()]
            region_id = np.full(n_cells, -1, dtype=np.int16)
            for i, scope in enumerate(scopes):
                region_id[scope] = i
            arrays["region_id"] = region_id
        else:
            scopes = [np.ones(n_cells, dtype=bool)]
        rank = rank_cells(scores, eligible, scopes)
        arrays["rank"] = rank

        if self.feature_sample_fraction is not None:
            sample_idx, sample_is_top, sample_prob = self._sample_cells(
                rank, eligible, selected, scopes
            )
            arrays["sample_cell_idx"] = sample_idx
            arrays["sample_is_top"] = sample_is_top
            arrays["sample_inclusion_prob"] = sample_prob
            arrays["features_input"] = features_input[:, sample_idx]
            if "features_raw" in arrays:
                arrays["features_raw"] = arrays["features_raw"][:, sample_idx]

        if not self._cells_written:
            coords = env.sdms._coords
            cells: dict[str, Any] = {
                "coords_row": np.asarray(coords[0]),
                "coords_col": np.asarray(coords[1]),
                "grid_shape": np.asarray(env.sdms._data_shape[1:]),
                "feature_names": feature_names,
            }
            if region_names is not None:
                cells["region_names"] = region_names
            self._save(self.out_dir / self.CELLS_FILE, **cells)
            self._cells_written = True

        step_file = self.out_dir / f"step_{t:03d}_u{update_idx:02d}.npz"
        self._save(step_file, **arrays)
        self.files.append(step_file)
        self._recorded_steps.add(t)
        self._n_files += 1

    def finalize(self) -> None:
        """Log a summary and warn about requested steps with no policy decision."""
        logger.info("Saved %d inference records to %s", self._n_files, self.out_dir)
        if self.steps is not None:
            missing = sorted(self.steps - self._recorded_steps)
            if missing:
                logger.warning(
                    "No policy decision (budget exhausted or step out of range) at steps: %s",
                    missing,
                )
