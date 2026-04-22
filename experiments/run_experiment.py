"""
experiments/run_experiment.py

Single-run experiment engine — trains one condition of the ablation matrix
and logs everything to wandb.

One call to run_experiment() covers:
    1. Agent construction (N TinyNets, same seed → fair comparison)
    2. Trainer construction (SwarmTrainer or IndependentEnsembleTrainer)
    3. Training loop with per-epoch wandb logging
    4. CKA computation at checkpoints → logged as heatmap images + scalars
    5. Final test evaluation
    6. Loss landscape computation for one representative agent
    7. Checkpoint saving

Usage
-----
Local dev (offline wandb, small subset):
    from experiments.run_experiment import ExperimentConfig, run_experiment
    from swarm.trainer import SwarmConfig

    cfg = ExperimentConfig(
        swarm=SwarmConfig(alpha=0.3, beta=0.5, gamma=0.1, k=3),
        epochs=10,
        n_agents=4,
        subset_size=5000,
        wandb_mode='offline',
    )
    results = run_experiment(cfg)

Colab A100 (online wandb, full dataset):
    cfg = ExperimentConfig(
        swarm=SwarmConfig(alpha=0.3, beta=0.5, gamma=0.1, k=3),
        epochs=50,
        n_agents=10,
        subset_size=None,
        device='cuda',
        wandb_mode='online',
    )
    results = run_experiment(cfg)
"""

import copy
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use('Agg')   # non-interactive — safe for Colab and headless runs
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import wandb
from tqdm.auto import tqdm

from baselines.single_trainer import (
    AgentTrainer, EarlyStopping, EnsembleTrainer, IndependentEnsembleTrainer, TrainingConfig
)
from swarm.trainer import SwarmConfig, SwarmTrainer
from models.cnn import TinyNet
from data.cifar import get_cifar10_loaders, get_probe_loader as get_cifar10_probe
from data.mnist import get_mnist_loaders, get_probe_loader as get_mnist_probe
from metrics.cka import CKATracker
from visualization.plots import (
    plot_cka_matrix, plot_diversity_curves,
    plot_training_curves, plot_agent_distances,
)
from visualization.loss_landscape import compute_loss_grid, plot_loss_landscape, plot_agent_pca


# ---------------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------------

@dataclass
class ExperimentConfig:
    """
    Full configuration for one experiment run.

    Parameters
    ----------
    swarm : SwarmConfig
        Controls which swarm rules are active and at what strength.
        SwarmConfig(alpha=0, beta=0, gamma=0) → baseline (independent ensemble).

    n_agents : int
        Number of agents in the ensemble. Default 10.

    epochs : int
        Training epochs. Default 30 for local dev; 50+ for Colab.

    batch_size : int
        Mini-batch size. Default 128.

    subset_size : int or None
        Training subset size. None = full CIFAR-10 (50k).
        Default 5000 for local dev.

    device : str
        'cpu', 'cuda', or 'mps'. Default 'cpu'.

    seed : int
        RNG seed for agent initialization and data splitting.
        All agents share the same seed base so differences between runs
        come only from the swarm rules, not from random initialization.

    lr : float
        Learning rate. Default 1e-3.

    weight_decay : float
        L2 regularization. Default 1e-4.

    warm_start_steps : int
        Steps before swarm rules activate. Default 0.

    cka_interval : int
        Compute CKA every this many epochs. Default 5.

    landscape_at_end : bool
        Compute loss landscape for agent 0 at the end of training.
        Expensive on CPU — set False for quick local runs.

    landscape_grid_size : int
        Number of points along each axis of the loss grid. 51 for final runs,
        11 or 15 for quick debugging.

    landscape_alpha_range : tuple[float, float]
        (min, max) perturbation range for both landscape axes. Use a tight
        range like (-0.5, 0.5) for fast debugging, (-2.0, 2.0) for full runs.

    checkpoint_dir : str or Path
        Where to save model checkpoints. Default 'experiments/checkpoints'.

    wandb_project : str
        W&B project name.

    wandb_mode : str
        'online', 'offline', or 'disabled'.
        Use 'offline' for local dev, 'online' for Colab A100 runs.

    dataset : str
        Which dataset to use. Either 'cifar10' or 'mnist'.
        Default 'cifar10'. Use 'mnist' for fast hyperparameter sweeps.

    num_workers : int
        DataLoader worker processes. Default 0.
        num_workers > 0 causes Python 3.12 multiprocessing cleanup errors on
        Colab, and the benefit is negligible for a small model like TinyNet
        where the GPU is the bottleneck, not data loading.

    run_name : str or None
        Human-readable W&B run name. Auto-generated from SwarmConfig if None.
    """
    swarm:             SwarmConfig  = field(default_factory=SwarmConfig)
    dataset:           str          = 'cifar10'
    num_workers:       int          = 0
    n_agents:          int          = 10
    epochs:            int          = 30
    batch_size:        int          = 512
    subset_size:       Optional[int]= 5000
    device:            str          = 'cpu'
    seed:              int          = 42
    lr:                float        = 1e-3
    weight_decay:      float        = 1e-4
    warm_start_steps:  int          = 0
    cka_interval:      int          = 5
    landscape_at_end:        bool                    = False
    landscape_grid_size:     int                     = 51
    landscape_alpha_range:   tuple[float, float]     = (-2.0, 2.0)
    checkpoint_dir:    Path         = Path('experiments/checkpoints')
    wandb_project:     str          = 'swarm-optimization'
    wandb_mode:        str          = 'offline'
    wandb_group:       Optional[str]= None
    run_name:          Optional[str]= None

    def __post_init__(self) -> None:
        self.checkpoint_dir = Path(self.checkpoint_dir)
        if self.run_name is None:
            baseline = not self.swarm.any_rule_active
            self.run_name = 'baseline' if baseline else self.swarm.label


# ---------------------------------------------------------------------------
# Agent factory
# ---------------------------------------------------------------------------

def build_agents(n: int, config: TrainingConfig, seed: int) -> list[AgentTrainer]:
    """
    Build N TinyNet agents, each initialized with a different seed
    derived from the base seed.

    Using seed + i ensures agents start at different points in weight
    space (genuine diversity from the start) while remaining reproducible.
    """
    agents = []
    for i in range(n):
        torch.manual_seed(seed + i)
        model = TinyNet()
        agents.append(AgentTrainer(model, config))
    return agents


# ---------------------------------------------------------------------------
# Main experiment function
# ---------------------------------------------------------------------------

def run_experiment(cfg: ExperimentConfig) -> dict:
    """
    Train one experiment condition end-to-end and return results.

    Parameters
    ----------
    cfg : ExperimentConfig

    Returns
    -------
    dict with keys:
        'history'       : list[dict] — per-epoch training metrics
        'test_metrics'  : dict — final test set evaluation
        'cka_history'   : list[dict] — CKA tracker history
        'config'        : ExperimentConfig
    """
    # ── 1. wandb init ────────────────────────────────────────────────────
    wandb.init(
        project = cfg.wandb_project,
        name    = cfg.run_name,
        group   = cfg.wandb_group,
        mode    = cfg.wandb_mode,
        config  = {
            'dataset':          cfg.dataset,
            'n_agents':         cfg.n_agents,
            'epochs':           cfg.epochs,
            'batch_size':       cfg.batch_size,
            'subset_size':      cfg.subset_size,
            'lr':               cfg.lr,
            'weight_decay':     cfg.weight_decay,
            'warm_start_steps': cfg.warm_start_steps,
            'seed':             cfg.seed,
            'alpha':            cfg.swarm.alpha,
            'beta':             cfg.swarm.beta,
            'gamma':            cfg.swarm.gamma,
            'k':                cfg.swarm.k,
            'update_interval':  cfg.swarm.update_interval,
        },
    )

    # ── 2. Data ───────────────────────────────────────────────────────────
    if cfg.dataset == 'mnist':
        train_loader, val_loader, test_loader = get_mnist_loaders(
            batch_size      = cfg.batch_size,
            subset_size     = cfg.subset_size,
            seed            = cfg.seed,
            num_workers     = cfg.num_workers,
            expand_channels = True,
        )
        probe_loader = get_mnist_probe(n_samples=512, batch_size=512)
    else:
        train_loader, val_loader, test_loader = get_cifar10_loaders(
            batch_size  = cfg.batch_size,
            subset_size = cfg.subset_size,
            seed        = cfg.seed,
            num_workers = cfg.num_workers,
        )
        probe_loader = get_cifar10_probe(n_samples=512, batch_size=512)

    # ── 3. Agents + trainer ───────────────────────────────────────────────
    train_cfg = TrainingConfig(
        lr               = cfg.lr,
        weight_decay     = cfg.weight_decay,
        warm_start_steps = cfg.warm_start_steps,
        device           = cfg.device,
        seed             = cfg.seed,
    )
    agents = build_agents(cfg.n_agents, train_cfg, seed=cfg.seed)

    criterion = nn.CrossEntropyLoss()

    if cfg.swarm.any_rule_active:
        trainer: EnsembleTrainer = SwarmTrainer(agents, criterion, cfg.swarm)
    else:
        trainer = IndependentEnsembleTrainer(agents, criterion)

    print(f'\nRun: {cfg.run_name}')
    print(f'Trainer: {trainer}')
    print(f'Agents: {cfg.n_agents} × TinyNet ({agents[0].model.param_count():,} params each)')
    print(f'Train batches: {len(train_loader)}  Val batches: {len(val_loader)}\n')

    # ── 4. CKA tracker ────────────────────────────────────────────────────
    cka_tracker = CKATracker(agents, probe_loader)

    # ── 5. Training loop ──────────────────────────────────────────────────
    early_stopping = EarlyStopping(patience=5, min_delta=1e-3)

    # Snapshot param vectors each epoch for trajectory visualization.
    # Shape per entry: list of (D,) tensors, one per agent.
    param_snapshots: list[list] = []

    def epoch_callback(epoch: int, metrics: dict) -> bool:
        """
        Called by EnsembleTrainer.train() at the end of each epoch.
        Returns True to trigger early stopping, False to continue.
        """

        # Per-agent and mean training loss
        log_dict: dict = {
            'train/mean_loss': metrics['mean_loss'],
            'epoch': epoch,
        }
        for i, loss in enumerate(metrics['agent_losses']):
            log_dict[f'train/agent_{i}_loss'] = loss

        # Parameter-space diversity (mean pairwise L2 distance)
        if isinstance(trainer, SwarmTrainer):
            dist_mat   = trainer.inter_agent_distances()
            mask       = ~torch.eye(cfg.n_agents, dtype=torch.bool)
            mean_dist  = dist_mat[mask].mean().item()
            log_dict['diversity/mean_param_distance'] = mean_dist

        # Validation loss + accuracy + F1
        val_metrics = trainer.evaluate(val_loader)
        log_dict['val/mean_loss']          = val_metrics['mean_loss']
        log_dict['val/best_loss']          = val_metrics['best_loss']
        log_dict['val/ensemble_loss']      = val_metrics['ensemble_loss']
        log_dict['val/mean_acc']           = val_metrics['mean_acc']
        log_dict['val/best_acc']           = val_metrics['best_acc']
        log_dict['val/ensemble_acc']       = val_metrics['ensemble_acc']
        log_dict['val/ensemble_f1']        = val_metrics['ensemble_f1']
        # Generalization gap: positive = overfitting, negative = underfitting.
        # Useful as a sweep metric — minimizing this selects configs that
        # generalize rather than just fitting training data.
        log_dict['val/generalization_gap'] = (
            val_metrics['ensemble_loss'] - metrics['mean_loss']
        )

        # CKA checkpoint
        if epoch % cfg.cka_interval == 0:
            cka_results = cka_tracker.compute(epoch)
            for layer, stats in cka_results.items():
                log_dict[f'cka/{layer}/mean_sim'] = stats['mean_sim']
                log_dict[f'cka/{layer}/min_sim']  = stats['min_sim']
                log_dict[f'cka/{layer}/max_sim']  = stats['max_sim']

                # Log CKA matrix as an image
                fig = plot_cka_matrix(stats['matrix'], layer=layer, epoch=epoch)
                log_dict[f'cka/{layer}/matrix'] = wandb.Image(fig)
                plt.close(fig)

        # Diversity curve image (logged every cka_interval)
        if epoch % cfg.cka_interval == 0 and len(cka_tracker.history) > 1:
            fig = plot_diversity_curves(cka_tracker.history)
            log_dict['cka/diversity_curves'] = wandb.Image(fig)
            plt.close(fig)

        wandb.log(log_dict, step=epoch)

        # Console progress — tqdm.write keeps output above the progress bars
        tqdm.write(
            f'Epoch {epoch:3d}  '
            f'train={metrics["mean_loss"]:.4f}  '
            f'val={val_metrics["mean_loss"]:.4f}  '
            f'ens={val_metrics["ensemble_loss"]:.4f}  '
            f'acc={val_metrics["ensemble_acc"]:.3f}'
        )

        # Snapshot weights for trajectory visualization
        param_snapshots.append([a.param_vector().cpu().clone() for a in agents])

        # Early stopping — monitor ensemble val loss
        return early_stopping.step(val_metrics['ensemble_loss'])

    history = trainer.train(train_loader, epochs=cfg.epochs, callback=epoch_callback)

    # ── 6. Final training curves image ───────────────────────────────────
    fig = plot_training_curves(history)
    wandb.log({'train/loss_curves': wandb.Image(fig)})
    plt.close(fig)

    # ── 7. Final test evaluation ─────────────────────────────────────────
    test_metrics = trainer.evaluate(test_loader)
    wandb.log({
        'test/mean_loss':     test_metrics['mean_loss'],
        'test/best_loss':     test_metrics['best_loss'],
        'test/ensemble_loss': test_metrics['ensemble_loss'],
        'test/mean_acc':      test_metrics['mean_acc'],
        'test/best_acc':      test_metrics['best_acc'],
        'test/ensemble_acc':  test_metrics['ensemble_acc'],
        'test/ensemble_f1':   test_metrics['ensemble_f1'],
        'test/diversity':     test_metrics['diversity'],
    })
    print(f'\nTest  ensemble={test_metrics["ensemble_loss"]:.4f}  '
          f'acc={test_metrics["ensemble_acc"]:.3f}  '
          f'best={test_metrics["best_loss"]:.4f}  '
          f'diversity={test_metrics["diversity"]:.4f}')

    # ── 8. Final parameter-space distance heatmap ─────────────────────────
    if isinstance(trainer, SwarmTrainer):
        dist_mat = trainer.inter_agent_distances()
        fig = plot_agent_distances(dist_mat, title=f'Final Param Distances — {cfg.run_name}')
        wandb.log({'final/param_distances': wandb.Image(fig)})
        plt.close(fig)

    # ── 9. Loss landscape (optional, expensive) ──────────────────────────
    cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if cfg.landscape_at_end:
        # ── Single-agent landscape (random filter-normalized directions) ──
        print('\nComputing loss landscape for agent 0...')
        grid = compute_loss_grid(
            agent       = agents[0],
            criterion   = criterion,
            loader      = probe_loader,
            grid_size   = cfg.landscape_grid_size,
            alpha_range = cfg.landscape_alpha_range,
        )
        fig = plot_loss_landscape(
            grid,
            title = f'Loss Landscape — {cfg.run_name}',
        )
        wandb.log({'final/loss_landscape': wandb.Image(fig)})
        landscape_path = cfg.checkpoint_dir / f'{cfg.run_name}_landscape.png'
        fig.savefig(landscape_path, dpi=150, bbox_inches='tight')
        print(f'Loss landscape saved → {landscape_path}')
        plt.close(fig)

        # ── Multi-agent PCA plot ──────────────────────────────────────────
        print('Computing multi-agent PCA plot...')
        fig = plot_agent_pca(
            agents          = agents,
            criterion       = criterion,
            loader          = probe_loader,
            param_snapshots = param_snapshots,
            agent_labels    = [f'A{i}' for i in range(cfg.n_agents)],
            grid_size       = cfg.landscape_grid_size,
            title           = f'Agent PCA — {cfg.run_name}',
        )
        wandb.log({'final/agent_pca': wandb.Image(fig)})
        pca_path = cfg.checkpoint_dir / f'{cfg.run_name}_pca.png'
        fig.savefig(pca_path, dpi=150, bbox_inches='tight')
        print(f'Agent PCA saved → {pca_path}')
        plt.close(fig)

    # ── 10. Save checkpoints ──────────────────────────────────────────────
    ckpt_path = cfg.checkpoint_dir / f'{cfg.run_name}.pt'
    torch.save(
        {
            'run_name':     cfg.run_name,
            'swarm_config': cfg.swarm,
            'agent_states': [a.model.state_dict() for a in agents],
            'history':      history,
            'test_metrics': test_metrics,
        },
        ckpt_path,
    )
    print(f'Checkpoint saved → {ckpt_path}')

    wandb.finish()

    return {
        'history':         history,
        'test_metrics':    test_metrics,
        'cka_history':     cka_tracker.history,
        'config':          cfg,
        'agents':          agents,
        'param_snapshots': param_snapshots,
        'criterion':       criterion,
        'probe_loader':    probe_loader,
    }
