"""
experiments/ablation.py

Ablation matrix runner — sweeps all 8 combinations of the three swarm rules
(α, β, γ each enabled or disabled) and collects results for comparison.

Ablation matrix (2^3 = 8 conditions)
--------------------------------------
    Condition          α      β      γ     label
    ─────────────────────────────────────────────
    Baseline           0      0      0     baseline
    Alignment only     α      0      0     alignment
    Separation only    0      β      0     separation
    Cohesion only      0      0      γ     cohesion
    Align + Sep        α      β      0     align_sep
    Align + Coh        α      0      γ     align_coh
    Sep + Coh          0      β      γ     sep_coh
    Full swarm         α      β      γ     full_swarm
    ─────────────────────────────────────────────

Usage
-----
Local dev (quick smoke test, 4 agents, 5 epochs):
    from experiments.ablation import run_ablation, AblationConfig
    results = run_ablation(AblationConfig(epochs=5, n_agents=4, subset_size=5000))

Colab A100 (full run):
    results = run_ablation(AblationConfig(
        epochs=50, n_agents=10, subset_size=None,
        device='cuda', wandb_mode='online',
    ))

Results
-------
run_ablation() returns a dict mapping condition label → experiment results dict.
Use compare_conditions() to print a summary table.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from swarm.trainer import SwarmConfig
from experiments.run_experiment import ExperimentConfig, run_experiment


# ---------------------------------------------------------------------------
# Ablation configuration
# ---------------------------------------------------------------------------

@dataclass
class AblationConfig:
    """
    Shared settings applied to every condition in the ablation sweep.

    Rule strengths
    --------------
    alpha, beta, gamma define the non-zero values used when a rule is
    active. Conditions with that rule disabled use 0.

    These defaults are a reasonable starting point:
        alpha = 0.3   moderate gradient blending
        beta  = 0.5   separation strength → equilibrium dist = β/γ = 5.0
        gamma = 0.1   cohesion pull
        k     = 3     each agent interacts with 3 nearest neighbors
    """
    # Rule strengths (used when rule is active)
    alpha: float = 0.3
    beta:  float = 0.5
    gamma: float = 0.1
    k:     int   = 3

    # Shared training settings
    n_agents:         int           = 10
    epochs:           int           = 30
    batch_size:       int           = 512
    subset_size:      Optional[int] = 5000
    device:           str           = 'cpu'
    seed:             int           = 42
    lr:               float         = 1e-3
    weight_decay:     float         = 1e-4
    warm_start_steps: int           = 0
    cka_interval:     int           = 5
    landscape_at_end: bool          = False
    checkpoint_dir:   Path          = Path('experiments/checkpoints')
    wandb_project:    str           = 'swarm-optimization'
    wandb_mode:       str           = 'offline'

    # Subset of conditions to run — None means all 8
    conditions: Optional[list[str]] = None

    def __post_init__(self) -> None:
        # k must be < n_agents — clamp so smoke tests with small n_agents
        # don't crash. k = n_agents - 1 is full connectivity.
        max_k = self.n_agents - 1
        if self.k > max_k:
            self.k = max_k


# ---------------------------------------------------------------------------
# Condition definitions
# ---------------------------------------------------------------------------

def _build_conditions(cfg: AblationConfig) -> dict[str, SwarmConfig]:
    """
    Return all 8 ablation conditions as a dict mapping label → SwarmConfig.

    Each condition enables/disables rules by setting their coefficient to
    the configured value or 0.
    """
    a, b, g, k = cfg.alpha, cfg.beta, cfg.gamma, cfg.k

    return {
        'baseline':   SwarmConfig(alpha=0, beta=0, gamma=0, k=k),
        'alignment':  SwarmConfig(alpha=a, beta=0, gamma=0, k=k),
        'separation': SwarmConfig(alpha=0, beta=b, gamma=0, k=k),
        'cohesion':   SwarmConfig(alpha=0, beta=0, gamma=g, k=k),
        'align_sep':  SwarmConfig(alpha=a, beta=b, gamma=0, k=k),
        'align_coh':  SwarmConfig(alpha=a, beta=0, gamma=g, k=k),
        'sep_coh':    SwarmConfig(alpha=0, beta=b, gamma=g, k=k),
        'full_swarm': SwarmConfig(alpha=a, beta=b, gamma=g, k=k),
    }


# ---------------------------------------------------------------------------
# Ablation runner
# ---------------------------------------------------------------------------

def run_ablation(cfg: AblationConfig) -> dict[str, dict]:
    """
    Run all (or a subset of) ablation conditions sequentially.

    Each condition is a fully independent run with its own wandb log,
    checkpoint, and results dict.

    Parameters
    ----------
    cfg : AblationConfig

    Returns
    -------
    dict mapping condition_label → results dict from run_experiment()
    Each results dict contains: 'history', 'test_metrics', 'cka_history', 'config'
    """
    conditions = _build_conditions(cfg)

    # Filter to requested subset if specified
    if cfg.conditions is not None:
        unknown = set(cfg.conditions) - set(conditions)
        if unknown:
            raise ValueError(
                f"Unknown condition(s): {unknown}. "
                f"Valid: {list(conditions.keys())}"
            )
        conditions = {k: v for k, v in conditions.items() if k in cfg.conditions}

    print(f'Running {len(conditions)} ablation condition(s): {list(conditions.keys())}')
    print('=' * 60)

    all_results: dict[str, dict] = {}

    for label, swarm_cfg in conditions.items():
        print(f'\n{"─"*60}')
        print(f'Condition: {label}  '
              f'(α={swarm_cfg.alpha}, β={swarm_cfg.beta}, γ={swarm_cfg.gamma})')
        print(f'{"─"*60}')

        exp_cfg = ExperimentConfig(
            swarm            = swarm_cfg,
            run_name         = label,
            n_agents         = cfg.n_agents,
            epochs           = cfg.epochs,
            batch_size       = cfg.batch_size,
            subset_size      = cfg.subset_size,
            device           = cfg.device,
            seed             = cfg.seed,
            lr               = cfg.lr,
            weight_decay     = cfg.weight_decay,
            warm_start_steps = cfg.warm_start_steps,
            cka_interval     = cfg.cka_interval,
            landscape_at_end = cfg.landscape_at_end,
            checkpoint_dir   = cfg.checkpoint_dir,
            wandb_project    = cfg.wandb_project,
            wandb_mode       = cfg.wandb_mode,
        )

        results = run_experiment(exp_cfg)
        all_results[label] = results

    print(f'\n{"="*60}')
    print('Ablation complete.')
    compare_conditions(all_results)

    return all_results


# ---------------------------------------------------------------------------
# Results comparison
# ---------------------------------------------------------------------------

def compare_conditions(results: dict[str, dict]) -> None:
    """
    Print a summary comparison table across all ablation conditions.

    Columns: condition, ensemble test loss, best agent loss,
             final diversity, final gap CKA mean similarity.
    """
    print(f'\n{"Condition":<14} {"Ens Loss":>10} {"Ens Acc":>10} '
          f'{"Best Acc":>10} {"Diversity":>10} {"GAP CKA":>10}')
    print('─' * 68)

    for label, res in results.items():
        tm       = res['test_metrics']
        ens_loss = tm['ensemble_loss']
        ens_acc  = tm['ensemble_acc']
        best_acc = tm['best_acc']
        div      = tm['diversity']

        # Final CKA at gap layer (last recorded checkpoint)
        gap_cka = '—'
        if res['cka_history']:
            last = res['cka_history'][-1]
            if 'gap' in last:
                gap_cka = f'{last["gap"]["mean_sim"]:.3f}'

        print(
            f'{label:<14} {ens_loss:>10.4f} {ens_acc:>10.3f} '
            f'{best_acc:>10.3f} {div:>10.2f} {gap_cka:>10}'
        )
