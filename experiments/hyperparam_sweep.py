"""
experiments/hyperparam_sweep.py

Bayesian optimization sweep for full_swarm rule strengths (alpha, beta, gamma)
on CIFAR-10 with all other parameters fixed.

Designed to run ONE trial per agent invocation (HPC-safe).
"""

from pathlib import Path
from typing import Optional

import wandb

from experiments.run_experiment import ExperimentConfig, run_experiment
from swarm.trainer import SwarmConfig


# ---------------------------------------------------------------------------
# Search space definition (FULL SWARM ONLY)
# ---------------------------------------------------------------------------

_FULL_SWARM_PARAMS = {
    'alpha': {'min': 0.1, 'max': 0.9},
    'beta':  {'min': 0.1, 'max': 0.9},
    'gamma': {'min': 0.1, 'max': 0.9},
}

ENTITY_NAME  = "blakedehaas-auroral-precipitation-ml"
PROJECT_NAME = "multi-agent-ml"


# ---------------------------------------------------------------------------
# Sweep creation
# ---------------------------------------------------------------------------

def create_sweep(
    method:      str = 'bayes',
    n_trials:    int = 64,
    metric_name: str = 'val/generalization_gap',
    metric_goal: str = 'minimize',
) -> str:
    """
    Register a Bayesian optimization sweep for full_swarm (alpha, beta, gamma).

    Call ONCE per sweep.
    """

    sweep_config = {
        'method': method,
        'metric': {
            'name': metric_name,
            'goal': metric_goal,
        },
        'parameters': _FULL_SWARM_PARAMS,
        'run_cap': n_trials,
    }

    sweep_id = wandb.sweep(
        sweep_config,
        project=PROJECT_NAME,
        entity=ENTITY_NAME,
    )

    print(
        f"View sweep at: https://wandb.ai/"
        f"{ENTITY_NAME}/{PROJECT_NAME}/sweeps/{sweep_id.split('/')[-1]}"
    )
    return sweep_id


# ---------------------------------------------------------------------------
# Sweep agent
# ---------------------------------------------------------------------------

def run_sweep_agent(
    sweep_id:    str,
    n_agents:    int          = 12,
    epochs:      int          = 50,
    subset_size: Optional[int]= None,
    device:      str          = 'cuda',
    seed:        int          = 42,
    cka_interval:int          = 5,
    checkpoint_dir: Path     = Path('experiments/checkpoints/sweep'),
    count:       Optional[int] = None,
) -> None:
    """
    Launch a sweep agent that runs EXACTLY `count` trials.

    For HPC usage:
        count=1 ensures no trial is cut short by walltime.
    """

    def _trial() -> None:
        with wandb.init() as run:
            wc = run.config

            swarm_cfg = SwarmConfig(
                alpha = float(wc.alpha),
                beta  = float(wc.beta),
                gamma = float(wc.gamma),
                k     = 4,   # fixed
            )

            exp_cfg = ExperimentConfig(
                swarm            = swarm_cfg,
                run_name         = f'full_swarm_{run.id}',
                n_agents         = n_agents,
                epochs           = epochs,
                batch_size       = 512,
                subset_size      = subset_size,
                device           = device,
                seed             = seed,
                lr               = 1e-3,
                weight_decay     = 1e-4,
                cka_interval     = cka_interval,
                landscape_at_end = False,
                checkpoint_dir   = checkpoint_dir,
                wandb_project    = PROJECT_NAME,
                wandb_mode       = 'online',
            )

            run_experiment(exp_cfg)


    wandb.agent(
        sweep_id,
        function=_trial,
    )
