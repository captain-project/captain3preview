#!/usr/bin/env python
"""Example: Run Reward Calibration

Calibrates reward-term scales once, always using the *global* agent
configuration (regardless of whatever USE_REGIONAL_AGENTS/REGION_ID is
currently set in train_policy.py), and saves the result to
reward_calibration.json.

Why global-only: calibration derives per-reward-term multipliers from probe
episodes, and those multipliers depend on how many cells get protected
during the probes. If every agent mode calibrated itself independently, a
global run and a regional run would end up with different multipliers and
their reward values would not be comparable. Running calibration once here
(reusing train_policy.py's own environment-building code, forced to
USE_REGIONAL_AGENTS=False, REGION_ID=None) and having train_policy.py always
load this file keeps reward scales consistent across all modes.

Run this once before training in any mode; re-run only if the reward terms,
weights, or dataset change.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_policy as tp  # noqa: E402

N_PROBES = 5  # Number of perturbations for reward calibration


def main():
    print("=" * 60)
    print("CAPTAIN-3 Reward calibration (global agent)")
    print("=" * 60)

    tp.USE_REGIONAL_AGENTS = False  # calibration always runs against the global agent
    tp.REGION_ID = None
    episode_runners = [tp.create_episode_runner()]
    episode = episode_runners[0]
    print(f"  Grid: {episode.env.n_cells} cells, {episode.env.n_species} species")

    trainer = tp.cn.EvolStrategiesTrainer(
        episode_runners,
        initial_coeffs=episode.policy.get_flat_weights(),
        scheduler=tp.cn.LearningScheduler(initial_alpha=0.2, initial_sigma=0.3),
        epsilon_reward=0.5,
        n_perturbations=tp.N_PERTURBATIONS,
        seed=tp.SEED,
    )

    multipliers = trainer.get_reward_calibrated_weights(n_probes=N_PROBES, verbose=True)
    trainer.save_reward_calibration(multipliers, tp.CALIBRATION_FILE, verbose=True)
    print(f"Calibration saved to: {tp.CALIBRATION_FILE}")

    trainer.close()


if __name__ == "__main__":
    main()
