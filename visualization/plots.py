"""
visualization/plots.py

General-purpose diagnostic plots for the swarm optimization experiment.

Functions
---------
    plot_cka_matrix        — heatmap of pairwise CKA at one checkpoint
    plot_diversity_curves  — mean CKA over training time per layer
    plot_training_curves   — per-agent + ensemble loss over epochs
    plot_agent_distances   — pairwise parameter-space distance heatmap

All functions return a matplotlib Figure so they can be:
    - shown interactively  : plt.show()
    - saved to disk        : fig.savefig('out.png', dpi=150)
    - logged to wandb      : wandb.log({'plot': wandb.Image(fig)})
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# Shared style helpers
# ---------------------------------------------------------------------------

_LAYER_COLORS = {
    'block1': '#4C72B0',
    'block2': '#DD8452',
    'block3': '#55A868',
    'gap':    '#C44E52',
}


def _agent_labels(n: int) -> list[str]:
    """Default agent axis labels: A0, A1, ..."""
    return [f'A{i}' for i in range(n)]


# ---------------------------------------------------------------------------
# 1. CKA matrix heatmap
# ---------------------------------------------------------------------------

def plot_cka_matrix(
    matrix: Tensor,
    layer: str = '',
    epoch: int | None = None,
    agent_labels: list[str] | None = None,
    figsize: tuple[float, float] = (5, 4),
) -> plt.Figure:
    """
    Plot a single (N_agents × N_agents) CKA matrix as a heatmap.

    Color scale is fixed [0, 1] so plots from different checkpoints and
    conditions are directly comparable.

    Parameters
    ----------
    matrix : Tensor shape (N, N)
    layer : str
        Layer name for the title (e.g. 'gap', 'block3').
    epoch : int or None
        Checkpoint epoch — shown in title if provided.
    agent_labels : list[str] or None
        Axis tick labels. Defaults to ['A0', 'A1', ...].
    figsize : tuple

    Returns
    -------
    matplotlib Figure
    """
    n = matrix.shape[0]
    labels = agent_labels or _agent_labels(n)
    data   = matrix.numpy() if isinstance(matrix, Tensor) else np.array(matrix)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(data, vmin=0.0, vmax=1.0, cmap='viridis', aspect='auto')
    plt.colorbar(im, ax=ax, label='CKA similarity')

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)

    # Annotate each cell with its value
    for i in range(n):
        for j in range(n):
            color = 'white' if data[i, j] < 0.6 else 'black'
            ax.text(j, i, f'{data[i, j]:.2f}', ha='center', va='center',
                    fontsize=8, color=color)

    title = 'CKA Similarity'
    if layer:
        title += f' — {layer}'
    if epoch is not None:
        title += f' (epoch {epoch})'
    ax.set_title(title, fontsize=11)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 2. Diversity curves (mean CKA over time)
# ---------------------------------------------------------------------------

def plot_diversity_curves(
    history: list[dict],
    layers: list[str] | None = None,
    figsize: tuple[float, float] = (7, 4),
) -> plt.Figure:
    """
    Plot mean off-diagonal CKA over training epochs for each layer.

    Lower values = agents have more diverse representations.
    Use this to see when/if separation causes agents to diverge.

    Parameters
    ----------
    history : list of dicts
        CKATracker.history  — list of dicts keyed by layer name,
        each containing 'epoch' and per-layer {'mean_sim', ...} dicts.
    layers : list[str] or None
        Subset of layers to plot. Defaults to all layers in history.
    figsize : tuple

    Returns
    -------
    matplotlib Figure
    """
    if not history:
        raise ValueError("history is empty — run CKATracker.compute() first.")

    layers = layers or [k for k in history[0] if k != 'epoch']
    epochs = [entry['epoch'] for entry in history]

    fig, ax = plt.subplots(figsize=figsize)

    for layer in layers:
        mean_sims = [entry[layer]['mean_sim'] for entry in history]
        min_sims  = [entry[layer]['min_sim']  for entry in history]
        max_sims  = [entry[layer]['max_sim']  for entry in history]

        color = _LAYER_COLORS.get(layer, None)
        ax.plot(epochs, mean_sims, label=layer, color=color, linewidth=2)
        ax.fill_between(epochs, min_sims, max_sims,
                        alpha=0.15, color=color)

    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Mean pairwise CKA similarity', fontsize=11)
    ax.set_title('Representational Diversity Over Training', fontsize=12)
    ax.set_ylim(0.0, 1.05)
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.05))
    ax.grid(True, alpha=0.3)
    ax.legend(title='Layer', fontsize=9, title_fontsize=9)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 3. Training loss curves
# ---------------------------------------------------------------------------

def plot_training_curves(
    history: list[dict],
    agent_labels: list[str] | None = None,
    figsize: tuple[float, float] = (8, 4),
) -> plt.Figure:
    """
    Plot per-agent training loss and ensemble mean loss over epochs.

    Parameters
    ----------
    history : list of dicts
        EnsembleTrainer.train() return value. Each dict has:
            'epoch'        : int
            'agent_losses' : list[float]
            'mean_loss'    : float
    agent_labels : list[str] or None
    figsize : tuple

    Returns
    -------
    matplotlib Figure
    """
    if not history:
        raise ValueError("history is empty.")

    epochs       = [d['epoch'] for d in history]
    agent_losses = [d['agent_losses'] for d in history]  # (epochs, n_agents)
    mean_losses  = [d['mean_loss'] for d in history]

    n_agents = len(history[0]['agent_losses'])
    labels   = agent_labels or _agent_labels(n_agents)

    fig, ax = plt.subplots(figsize=figsize)

    # Per-agent curves (thin, semi-transparent)
    for i in range(n_agents):
        agent_curve = [step[i] for step in agent_losses]
        ax.plot(epochs, agent_curve, linewidth=1.0, alpha=0.45,
                label=labels[i])

    # Ensemble mean (thick, prominent)
    ax.plot(epochs, mean_losses, linewidth=2.5, color='black',
            linestyle='--', label='Ensemble mean', zorder=5)

    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Loss', fontsize=11)
    ax.set_title('Training Loss per Agent', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=2)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 4. Pairwise parameter-space distance heatmap
# ---------------------------------------------------------------------------

def plot_agent_distances(
    distance_matrix: Tensor,
    agent_labels: list[str] | None = None,
    title: str = 'Pairwise Parameter-Space Distances',
    figsize: tuple[float, float] = (5, 4),
) -> plt.Figure:
    """
    Plot the (N × N) pairwise L2 distance matrix in parameter space.

    Complements CKA: CKA measures representational similarity while this
    measures raw weight-space proximity. Together they show whether agents
    that are far in weight space are also functionally diverse.

    Parameters
    ----------
    distance_matrix : Tensor shape (N, N)
        From KNNTopology.pairwise_distances() or SwarmTrainer.inter_agent_distances().
    agent_labels : list[str] or None
    title : str
    figsize : tuple

    Returns
    -------
    matplotlib Figure
    """
    n    = distance_matrix.shape[0]
    data = distance_matrix.numpy() if isinstance(distance_matrix, Tensor) else np.array(distance_matrix)
    labels = agent_labels or _agent_labels(n)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(data, cmap='plasma', aspect='auto')
    plt.colorbar(im, ax=ax, label='L2 distance')

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)

    # Annotate cells
    vmax = data.max()
    for i in range(n):
        for j in range(n):
            color = 'white' if data[i, j] < vmax * 0.6 else 'black'
            ax.text(j, i, f'{data[i, j]:.1f}', ha='center', va='center',
                    fontsize=7, color=color)

    ax.set_title(title, fontsize=11)
    fig.tight_layout()
    return fig
