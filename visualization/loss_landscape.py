"""
visualization/loss_landscape.py

Filter-normalized 2D loss landscape visualization (Li et al. 2018).

Core idea
---------
Pick two random directions δ, η in parameter space and evaluate the loss
at a grid of points around the trained weights θ*:

    L(a, b) = loss(θ* + a*δ_hat + b*η_hat)

The key is filter normalization — each direction is scaled so that the
perturbation magnitude is proportional to the filter norms of the model.
This makes landscapes from different models and checkpoints comparable on
the same axes.

Without normalization, a model with large weights would appear to have a
wide flat basin simply because the random direction has small components
relative to its weights. Filter normalization removes this scale artifact.

Reference: Li et al. 2018 — "Visualizing the Loss Landscape of Neural Nets"
           https://arxiv.org/abs/1712.09913

Usage
-----
    # After training agent to θ*:
    grid = compute_loss_grid(
        agent     = trained_agent,
        criterion = nn.CrossEntropyLoss(),
        loader    = probe_loader,
        grid_size = 21,
        alpha_range = (-1.0, 1.0),
    )

    fig = plot_loss_landscape(grid, agents=[agent_a, agent_b])
    wandb.log({'landscape': wandb.Image(fig)})
"""

import copy

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from torch import Tensor
from torch.utils.data import DataLoader

from baselines.single_trainer import AgentTrainer
from models.cnn import TinyNet


# ---------------------------------------------------------------------------
# Filter normalization
# ---------------------------------------------------------------------------

def filter_normalize_direction(
    direction: Tensor,
    model: TinyNet,
) -> Tensor:
    """
    Scale a random direction vector so perturbation magnitude matches
    the filter norms of the model (Li et al. 2018, Section 4).

    For each filter f in layer l, the direction is rescaled:
        d_hat[l,f] = d[l,f] / ||d[l,f]|| * ||θ*[l,f]||

    This ensures a step of size 1.0 in the landscape corresponds to
    a perturbation of the same relative magnitude as the actual weights
    at every layer, making landscapes comparable across models.

    Parameters
    ----------
    direction : Tensor shape (D,)
        Flat random direction vector (same size as param_vector()).
    model : TinyNet
        The trained model whose filter norms we normalize against.

    Returns
    -------
    Tensor shape (D,) — filter-normalized direction.
    """
    normalized = direction.clone()
    offset = 0

    for param in model.parameters():
        numel = param.numel()
        chunk = normalized[offset: offset + numel].view_as(param)

        if param.dim() == 4:
            # Conv filter: shape (out_ch, in_ch, kH, kW)
            # Normalize each output filter independently
            for f in range(param.shape[0]):
                filter_dir    = chunk[f]
                filter_weight = param[f]
                dir_norm    = filter_dir.norm()
                weight_norm = filter_weight.norm()
                if dir_norm > 1e-10:
                    chunk[f] = filter_dir / dir_norm * weight_norm
        else:
            # Linear / bias / BN: normalize the whole tensor as one unit
            dir_norm    = chunk.norm()
            weight_norm = param.norm()
            if dir_norm > 1e-10:
                chunk.copy_(chunk / dir_norm * weight_norm)

        normalized[offset: offset + numel] = chunk.flatten()
        offset += numel

    return normalized


# ---------------------------------------------------------------------------
# Loss grid computation
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_loss_grid(
    agent: AgentTrainer,
    criterion: nn.Module,
    loader: DataLoader,
    grid_size: int = 21,
    alpha_range: tuple[float, float] = (-1.0, 1.0),
    beta_range:  tuple[float, float] | None = None,
    seed: int = 42,
) -> dict:
    """
    Evaluate loss on a 2D grid of perturbations around the agent's weights.

    The two directions are sampled randomly and then filter-normalized so
    the grid axes have physically meaningful scale.

    Parameters
    ----------
    agent : AgentTrainer
        Trained agent. Weights are NOT modified — a temporary copy is used
        for each grid point evaluation.
    criterion : nn.Module
        Loss function (e.g. nn.CrossEntropyLoss()).
    loader : DataLoader
        Probe loader — use get_probe_loader() for reproducibility.
        Smaller loaders (512 samples) make grid computation fast.
    grid_size : int
        Number of points along each axis. grid_size=21 → 21×21 = 441
        evaluations. Increase to 51 on Colab for smoother plots.
    alpha_range : tuple (min, max)
        Range of perturbation magnitude along first direction.
    beta_range : tuple or None
        Range along second direction. Defaults to same as alpha_range.
    seed : int
        RNG seed for direction sampling. Fix this to compare landscapes
        from different agents on the same axes.

    Returns
    -------
    dict with keys:
        'loss_grid'   : ndarray shape (grid_size, grid_size)
        'alpha_vals'  : ndarray shape (grid_size,)
        'beta_vals'   : ndarray shape (grid_size,)
        'theta_star'  : Tensor (D,) — the center point (agent's weights)
        'delta'       : Tensor (D,) — filter-normalized direction 1
        'eta'         : Tensor (D,) — filter-normalized direction 2
        'center_loss' : float — loss at θ* (a=0, b=0)
    """
    if not isinstance(agent.model, TinyNet):
        raise TypeError("compute_loss_grid requires a TinyNet agent.")

    beta_range = beta_range or alpha_range

    # ── Sample and filter-normalize two random directions ──────────────
    torch.manual_seed(seed)
    theta_star = agent.param_vector()                            # (D,) on agent.device
    D = theta_star.shape[0]

    delta_raw = torch.randn(D, device=agent.device)
    eta_raw   = torch.randn(D, device=agent.device)

    delta = filter_normalize_direction(delta_raw, agent.model)   # (D,)
    eta   = filter_normalize_direction(eta_raw,   agent.model)   # (D,)

    # ── Build grid axes ──────────────────────────────────────────────────
    alpha_vals = np.linspace(alpha_range[0], alpha_range[1], grid_size)
    beta_vals  = np.linspace(beta_range[0],  beta_range[1],  grid_size)
    loss_grid  = np.zeros((grid_size, grid_size))

    # ── Evaluate loss at each grid point ─────────────────────────────────
    # Work on a temporary copy of the model to avoid modifying the agent
    tmp_model = copy.deepcopy(agent.model).to(agent.device)
    tmp_model.eval()

    from torch.nn.utils import vector_to_parameters

    for i, a in enumerate(alpha_vals):
        for j, b in enumerate(beta_vals):
            # Perturb: θ* + a*δ + b*η
            perturbed = theta_star + a * delta + b * eta
            vector_to_parameters(perturbed.to(agent.device), tmp_model.parameters())

            # Evaluate loss over the probe loader
            total_loss = 0.0
            n_batches  = 0
            for X, y in loader:
                X, y = X.to(agent.device), y.to(agent.device)
                logits = tmp_model(X)
                total_loss += criterion(logits, y).item()
                n_batches  += 1

            loss_grid[i, j] = total_loss / max(n_batches, 1)

    # Restore original weights
    from torch.nn.utils import vector_to_parameters as v2p
    v2p(theta_star.to(agent.device), agent.model.parameters())

    center_loss = loss_grid[grid_size // 2, grid_size // 2]

    return {
        'loss_grid':   loss_grid,
        'alpha_vals':  alpha_vals,
        'beta_vals':   beta_vals,
        'theta_star':  theta_star,
        'delta':       delta,
        'eta':         eta,
        'center_loss': center_loss,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_loss_landscape(
    grid: dict,
    agents: list[AgentTrainer] | None = None,
    agent_labels: list[str] | None = None,
    title: str = 'Loss Landscape',
    log_scale: bool = True,
    figsize: tuple[float, float] = (16, 6),
    elev: float = 30.0,
    azim: float = -60.0,
) -> plt.Figure:
    """
    Plot the loss landscape as a side-by-side 3D surface and 2D contour map.

    The 3D surface gives an intuitive view of basin geometry. The 2D contour
    makes agent positions easier to read precisely. Both share the same data
    and color scale.

    Agent positions are projected onto the (δ, η) plane. On the 3D plot they
    are drawn as vertical stems so they are visible regardless of viewing angle.
    On the 2D plot they are drawn as labeled scatter points.

    Parameters
    ----------
    grid : dict
        Output of compute_loss_grid().
    agents : list[AgentTrainer] or None
        Agents to overlay. Agent 0 is the center (θ*).
    agent_labels : list[str] or None
    log_scale : bool
        Plot log(loss) — makes basin edges more visible.
    figsize : tuple
    elev : float
        Elevation angle of the 3D view in degrees.
    azim : float
        Azimuth angle of the 3D view in degrees.

    Returns
    -------
    matplotlib Figure
    """
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers 3D projection

    loss_grid  = grid['loss_grid']
    alpha_vals = grid['alpha_vals']
    beta_vals  = grid['beta_vals']
    theta_star = grid['theta_star']
    delta      = grid['delta']
    eta        = grid['eta']

    A, B = np.meshgrid(alpha_vals, beta_vals, indexing='ij')
    Z    = np.log(loss_grid + 1e-8) if log_scale else loss_grid
    Z_label = 'log(loss)' if log_scale else 'loss'
    z_floor = Z.min() - 0.05 * (Z.max() - Z.min())

    # ── Project agents onto (δ, η) plane ─────────────────────────────────
    agent_projections = []
    if agents:
        labels = agent_labels or [f'A{i}' for i in range(len(agents))]
        colors = plt.cm.tab10(np.linspace(0, 1, len(agents)))
        for i, agent in enumerate(agents):
            if i == 0:
                continue
            diff   = agent.param_vector().to(theta_star.device) - theta_star
            proj_a = (torch.dot(diff, delta) / (delta.norm() ** 2 + 1e-10)).item()
            proj_b = (torch.dot(diff, eta)   / (eta.norm()   ** 2 + 1e-10)).item()
            agent_projections.append((proj_a, proj_b, labels[i], colors[i]))

    fig = plt.figure(figsize=figsize)
    fig.suptitle(title, fontsize=13, y=1.01)

    # ── Left: 3D surface ─────────────────────────────────────────────────
    ax3d = fig.add_subplot(121, projection='3d')

    surf = ax3d.plot_surface(
        A, B, Z,
        cmap='coolwarm',
        alpha=0.88,
        linewidth=0,
        antialiased=True,
    )

    # Center θ* — use plot3D for clean circular marker
    z_center = Z[len(alpha_vals) // 2, len(beta_vals) // 2]
    ax3d.plot3D([0], [0], [z_center], 'o',
                color='white', markersize=8,
                markeredgecolor='black', markeredgewidth=1.5,
                label='θ* (center)', zorder=5)

    # Agents as vertical stems
    for proj_a, proj_b, label, color in agent_projections:
        ax3d.plot([proj_a, proj_a], [proj_b, proj_b],
                  [z_floor, z_floor + 0.03],
                  color=color, linewidth=2, zorder=6)
        ax3d.plot3D([proj_a], [proj_b], [z_floor + 0.03], 'o',
                    color=color, markersize=7,
                    markeredgecolor='black', markeredgewidth=0.8,
                    label=label, zorder=7)

    ax3d.set_xlabel('α (direction δ)', fontsize=9, labelpad=8)
    ax3d.set_ylabel('β (direction η)', fontsize=9, labelpad=8)
    ax3d.set_zlabel(Z_label, fontsize=9, labelpad=8)
    ax3d.set_title('3D Surface', fontsize=11)
    ax3d.view_init(elev=elev, azim=azim)
    ax3d.legend(fontsize=8, loc='upper left')

    # ── Right: 2D contour ────────────────────────────────────────────────
    ax2d = fig.add_subplot(122)

    contourf = ax2d.contourf(A, B, Z, levels=30, cmap='coolwarm')
    ax2d.contour(A, B, Z, levels=30, colors='k', linewidths=0.3, alpha=0.4)
    fig.colorbar(contourf, ax=ax2d, label=Z_label, shrink=0.85)

    # Center θ*
    ax2d.plot(0, 0, 'o', color='white', markersize=10,
              markeredgecolor='black', markeredgewidth=1.5,
              label='θ* (center)', zorder=5)

    # Agents
    for proj_a, proj_b, label, color in agent_projections:
        ax2d.plot(proj_a, proj_b, 'o', color=color, markersize=8,
                  markeredgecolor='black', markeredgewidth=0.8,
                  label=label, zorder=5)

    ax2d.set_xlabel('α (direction δ)', fontsize=9)
    ax2d.set_ylabel('β (direction η)', fontsize=9)
    ax2d.set_title('2D Contour', fontsize=11)
    ax2d.legend(fontsize=8, loc='upper left')

    fig.tight_layout()
    return fig
