"""
experiments/hyperparam_sweep.py

Weights & Biases hyperparameter sweep for swarm rule strengths.

Intended use
------------
Run AFTER the ablation matrix has identified which combination of rules
(α, β, γ) produces the best results. The sweep then finds the optimal
values for those strengths within that combination.

Workflow
--------
1. Pick a condition from the ablation (e.g. 'full_swarm' or 'sep_coh')
2. Define a search space around that condition's active parameters
3. Create a W&B sweep (one call, returns a sweep_id)
4. Launch an agent that pulls configs and runs experiments

On Colab A100:
    from experiments.hyperparam_sweep import create_sweep, run_sweep_agent

    # Step 1: create the sweep (run once)
    sweep_id = create_sweep(
        condition   = 'full_swarm',
        project     = 'swarm-optimization',
        method      = 'bayes',
        n_trials    = 30,
    )

    # Step 2: launch agent (can launch multiple in parallel on separate Colab cells)
    run_sweep_agent(sweep_id, project='swarm-optimization')

Sweep methods
-------------
    'bayes'  — Bayesian optimization. Learns from previous trials to focus
               on promising regions. Best for expensive runs (recommended).
    'random' — Random search. Good baseline, embarrassingly parallel.
    'grid'   — Exhaustive grid. Only practical for 1-2 parameters.

Search spaces
-------------
Each condition has a pre-defined search space that only includes its
active swarm rule parameters. For example, 'separation' only sweeps β —
sweeping α and γ when they're 0 would be wasteful. lr and k are fixed
constants (1e-3 and 4 respectively) and are not part of the search space.
"""

from pathlib import Path
from typing import Optional

import wandb

from experiments.run_experiment import ExperimentConfig, run_experiment
from swarm.trainer import SwarmConfig


# ---------------------------------------------------------------------------
# Search space definitions (one per ablation condition)
# ---------------------------------------------------------------------------

# Per-condition search spaces — only the active swarm rule parameters are swept.
# lr is fixed at 1e-3 (not swept): the previous sweep showed it is not a
# significant confound and sweeping it wastes trials on a known-good value.
# k is fixed at 4 — topology and k sweeps established this as optimal.
_CONDITION_PARAMS: dict[str, dict] = {
    'baseline': {},   # no swarm params to tune
    'alignment': {
        'alpha': {'min': 0.05, 'max': 0.8},
    },
    'separation': {
        'beta': {'min': 0.05, 'max': 2.0},
    },
    'cohesion': {
        'gamma': {'min': 0.01, 'max': 0.5},
    },
    'align_sep': {
        'alpha': {'min': 0.05, 'max': 0.8},
        'beta':  {'min': 0.05, 'max': 2.0},
    },
    'align_coh': {
        'alpha': {'min': 0.05, 'max': 0.8},
        'gamma': {'min': 0.01, 'max': 0.5},
    },
    'sep_coh': {
        'beta':  {'min': 0.05, 'max': 2.0},
        'gamma': {'min': 0.01, 'max': 0.5},
    },
    'full_swarm': {
        'alpha': {'min': 0.05, 'max': 0.8},
        'beta':  {'min': 0.05, 'max': 1.0},
        'gamma': {'min': 0.01, 'max': 1.0},
    },
}


# ---------------------------------------------------------------------------
# Sweep creation
# ---------------------------------------------------------------------------

def create_sweep(
    condition:   str,
    project:     str   = 'swarm-optimization',
    method:      str   = 'bayes',
    n_trials:    int   = 30,
    metric_name: str   = 'val/diversity_weighted_acc',
    metric_goal: str   = 'maximize',
) -> str:
    """
    Register a new sweep with W&B and return its sweep_id.

    Call this ONCE per sweep — it creates the sweep configuration on the
    W&B server. Then use the returned sweep_id with run_sweep_agent().

    Parameters
    ----------
    condition : str
        Which ablation condition to tune. Must be one of:
        baseline, alignment, separation, cohesion,
        align_sep, align_coh, sep_coh, full_swarm.
    project : str
        W&B project name.
    method : str
        Search strategy: 'bayes', 'random', or 'grid'.
    n_trials : int
        Maximum number of trials. W&B stops the sweep after this many runs.
        Ignored for 'grid' method (which runs all combinations).
    metric_name : str
        The W&B logged metric to optimize.
    metric_goal : str
        'minimize' or 'maximize'.

    Returns
    -------
    str — sweep_id (e.g. 'osro6012-university-of-colorado-boulder/swarm-optimization/abc123')
    """
    if condition not in _CONDITION_PARAMS:
        raise ValueError(
            f"Unknown condition '{condition}'. "
            f"Valid: {list(_CONDITION_PARAMS.keys())}"
        )

    sweep_config = {
        'method': method,
        'metric': {
            'name': metric_name,
            'goal': metric_goal,
        },
        'parameters': _CONDITION_PARAMS[condition],
        'run_cap': n_trials,
    }

    sweep_id = wandb.sweep(sweep_config, project=project)
    print(f'Sweep created: {sweep_id}')
    print(f'View at: https://wandb.ai/{project}/sweeps/{sweep_id.split("/")[-1]}')
    return sweep_id


# ---------------------------------------------------------------------------
# Sweep agent
# ---------------------------------------------------------------------------

def run_sweep_agent(
    sweep_id:    str,
    condition:   str,
    project:     str          = 'swarm-optimization',
    n_agents:    int          = 12,
    epochs:      int          = 30,
    subset_size: Optional[int]= None,
    device:      str          = 'cuda',
    seed:        int          = 42,
    cka_interval:int          = 5,
    checkpoint_dir: Path      = Path('experiments/checkpoints/sweep'),
    count:       Optional[int]= None,
) -> None:
    """
    Launch a sweep agent that pulls configs from W&B and runs experiments.

    Each agent run:
        1. W&B assigns a hyperparameter config from the sweep
        2. run_experiment() trains with that config
        3. Metrics are logged; W&B updates its model of the search space
        4. Repeat until `count` runs complete or sweep is finished

    Multiple agents can run in parallel (e.g. on separate Colab cells or
    separate machines) — they all pull from the same sweep queue.

    Parameters
    ----------
    sweep_id : str
        Returned by create_sweep().
    condition : str
        Must match the condition used in create_sweep().
    project : str
    n_agents : int
        Number of ensemble agents per trial.
    epochs : int
        Training epochs per trial.
    subset_size : int or None
        Training subset. None = full CIFAR-10.
    device : str
        'cuda' for Colab A100, 'cpu' for local testing.
    seed : int
        Fixed seed — keeps agent initialization identical across trials
        so differences come only from swarm rule strengths (alpha/beta/gamma).
        lr is fixed at 1e-3; k is fixed at 4.
    cka_interval : int
    checkpoint_dir : Path
    count : int or None
        Maximum runs for this agent process. None = run until sweep ends.
    """

    def _trial() -> None:
        """One sweep trial — called by wandb.agent for each config."""

        # W&B injects the sampled config via wandb.config after init
        with wandb.init() as run:
            wc = run.config   # sampled hyperparameters for this trial

            # Build SwarmConfig from sampled values — unsampled params
            # default to 0 (disabled) for the given condition
            swarm_cfg = _config_from_wandb(condition, wc)

            exp_cfg = ExperimentConfig(
                swarm            = swarm_cfg,
                run_name         = f'{condition}_{run.id}',
                n_agents         = n_agents,
                epochs           = epochs,
                batch_size       = 512,
                subset_size      = subset_size,
                device           = device,
                seed             = seed,
                lr               = 1e-3,   # fixed — not swept
                cka_interval     = cka_interval,
                landscape_at_end = False,   # too expensive during sweep
                checkpoint_dir   = checkpoint_dir,
                wandb_project    = project,
                wandb_mode       = 'online',
            )

            run_experiment(exp_cfg)

    wandb.agent(sweep_id, function=_trial, count=count)


# ---------------------------------------------------------------------------
# Internal: build SwarmConfig from wandb sampled config
# ---------------------------------------------------------------------------

def _config_from_wandb(condition: str, wc) -> SwarmConfig:
    """
    Build a SwarmConfig from a wandb.config object for the given condition.

    Parameters not in the search space for this condition default to 0
    (rule disabled), except for k which defaults to 3.
    """
    alpha = float(wc.get('alpha', 0.0))
    beta  = float(wc.get('beta',  0.0))
    gamma = float(wc.get('gamma', 0.0))
    # k fixed at 4 — established as optimal by topology and k sweeps.
    k     = 4

    return SwarmConfig(alpha=alpha, beta=beta, gamma=gamma, k=k)


# ---------------------------------------------------------------------------
# Quick local validation (no actual wandb sweep — just checks imports)
# ---------------------------------------------------------------------------

def validate_sweep_config(condition: str = 'full_swarm') -> None:
    """
    Verify the sweep config for a condition is well-formed without
    actually creating a W&B sweep. Useful for local sanity checks.
    """
    if condition not in _CONDITION_PARAMS:
        raise ValueError(f"Unknown condition: {condition}")

    params = _CONDITION_PARAMS[condition]
    print(f"Sweep config for '{condition}':")
    for name, spec in params.items():
        print(f"  {name}: {spec}")

    # Verify a mock wandb config produces a valid SwarmConfig
    class MockConfig:
        def get(self, key, default=None):
            specs = params.get(key)
            if specs is None:
                return default
            if 'values' in specs:
                return specs['values'][0]
            return (specs['min'] + specs['max']) / 2

    mock_wc     = MockConfig()
    swarm_cfg   = _config_from_wandb(condition, mock_wc)
    print(f"\nMock SwarmConfig: α={swarm_cfg.alpha:.3f}, β={swarm_cfg.beta:.3f}, "
          f"γ={swarm_cfg.gamma:.3f}, k={swarm_cfg.k}")
    print("Sweep config valid.")
