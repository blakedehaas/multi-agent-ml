"""
visualization/loss_landscape.py

Two complementary loss landscape visualizations.

1. Single-agent landscape (Li et al. 2018)
-------------------------------------------
Filter-normalized random directions around a single agent's final weights.
Rigorous and comparable across conditions — the axes mean the same thing
for every model so basin shapes can be directly compared.

    grid = compute_loss_grid(agent, criterion, loader, grid_size=51)
    fig  = plot_loss_landscape(grid, title='Loss Landscape — baseline')

2. Multi-agent PCA plot
------------------------
PCA directions derived from all agents' final positions. The loss surface
is computed using PC1/PC2 as axes, centered on the mean agent position.
Agent dots and per-agent training trails are overlaid. Visually informative
for showing swarm diversity and convergence behavior.

    fig = plot_agent_pca(
        agents, criterion, probe_loader, param_snapshots,
        title='Agent PCA — full_swarm',
    )

Reference: Li et al. 2018 — "Visualizing the Loss Landscape of Neural Nets"
           https://arxiv.org/abs/1712.09913
"""

import copy

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
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
            for f in range(param.shape[0]):
                filter_dir    = chunk[f]
                filter_weight = param[f]
                dir_norm    = filter_dir.norm()
                weight_norm = filter_weight.norm()
                if dir_norm > 1e-10:
                    chunk[f] = filter_dir / dir_norm * weight_norm
        else:
            # Linear / bias / BN: normalize as one unit
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
    center: Tensor | None = None,
    directions: tuple[Tensor, Tensor] | None = None,
) -> dict:
    """
    Evaluate loss on a 2D grid of perturbations around a center point.

    By default the center is the agent's final weights and the directions
    are random filter-normalized vectors (Li et al. 2018). Pass `center`
    and `directions` to override both — used by plot_agent_pca to build
    a grid around the mean agent position along PCA axes.

    Parameters
    ----------
    agent : AgentTrainer
        Used for the model architecture and device. Weights are not modified.
    criterion : nn.Module
    loader : DataLoader
        Probe loader — smaller is faster (512 samples is enough).
    grid_size : int
        Points per axis. 51 for final runs, 11-15 for debugging.
    alpha_range : tuple (min, max)
        Range along first direction.
    beta_range : tuple or None
        Range along second direction. Defaults to alpha_range.
    seed : int
        RNG seed for random direction sampling. Fix to compare conditions.
    center : Tensor (D,) or None
        Center point for the grid. Defaults to agent.param_vector().
    directions : (Tensor, Tensor) or None
        (delta, eta) directions. When provided, filter normalization and
        random sampling are skipped entirely.

    Returns
    -------
    dict with keys:
        'loss_grid'   : ndarray (grid_size, grid_size)
        'alpha_vals'  : ndarray (grid_size,)
        'beta_vals'   : ndarray (grid_size,)
        'theta_star'  : Tensor (D,) — center point
        'delta'       : Tensor (D,) — direction 1
        'eta'         : Tensor (D,) — direction 2
        'center_loss' : float
    """
    if not isinstance(agent.model, TinyNet):
        raise TypeError("compute_loss_grid requires a TinyNet agent.")

    beta_range = beta_range or alpha_range

    # ── Center and directions ─────────────────────────────────────────────
    if center is not None:
        theta_star = center.to(agent.device)
    else:
        theta_star = agent.param_vector()

    if directions is not None:
        delta, eta = directions[0].to(agent.device), directions[1].to(agent.device)
    else:
        D = theta_star.shape[0]
        torch.manual_seed(seed)
        delta = filter_normalize_direction(torch.randn(D, device=agent.device), agent.model)
        eta   = filter_normalize_direction(torch.randn(D, device=agent.device), agent.model)

    # ── Build grid ────────────────────────────────────────────────────────
    alpha_vals = np.linspace(alpha_range[0], alpha_range[1], grid_size)
    beta_vals  = np.linspace(beta_range[0],  beta_range[1],  grid_size)
    loss_grid  = np.zeros((grid_size, grid_size))

    tmp_model = copy.deepcopy(agent.model).to(agent.device)
    tmp_model.eval()

    from torch.nn.utils import vector_to_parameters

    for i, a in enumerate(alpha_vals):
        for j, b in enumerate(beta_vals):
            perturbed = theta_star + a * delta + b * eta
            vector_to_parameters(perturbed, tmp_model.parameters())

            total_loss, n_batches = 0.0, 0
            for X, y in loader:
                X, y = X.to(agent.device), y.to(agent.device)
                total_loss += criterion(tmp_model(X), y).item()
                n_batches  += 1

            loss_grid[i, j] = total_loss / max(n_batches, 1)

    # Restore original weights
    from torch.nn.utils import vector_to_parameters as v2p
    v2p(agent.param_vector(), agent.model.parameters())

    return {
        'loss_grid':   loss_grid,
        'alpha_vals':  alpha_vals,
        'beta_vals':   beta_vals,
        'theta_star':  theta_star,
        'delta':       delta,
        'eta':         eta,
        'center_loss': loss_grid[grid_size // 2, grid_size // 2],
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _draw_surface_and_contour(
    fig: plt.Figure,
    ax3d,
    ax2d,
    A: np.ndarray,
    B: np.ndarray,
    Z: np.ndarray,
    Z_label: str,
    elev: float,
    azim: float,
    xlabel: str,
    ylabel: str,
    vmin: float | None = None,
    vmax: float | None = None,
) -> object:
    """Draw 3D surface + 2D contour onto pre-created axes. Returns contourf."""
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    # When a shared scale is provided, pin the levels explicitly so that
    # the colorbar ticks span the full shared range on every subplot —
    # not just the data range of this particular condition.
    if vmin is not None and vmax is not None:
        levels_arg = np.linspace(vmin, vmax, 31)
    else:
        levels_arg = 30

    surf = ax3d.plot_surface(A, B, Z, cmap='coolwarm', alpha=0.88,
                             linewidth=0, antialiased=True,
                             vmin=vmin, vmax=vmax)
    ax3d.set_xlabel(xlabel, fontsize=9, labelpad=8)
    ax3d.set_ylabel(ylabel, fontsize=9, labelpad=8)
    ax3d.set_zlabel(Z_label, fontsize=9, labelpad=8)
    ax3d.view_init(elev=elev, azim=azim)

    cf = ax2d.contourf(A, B, Z, levels=levels_arg, cmap='coolwarm', vmin=vmin, vmax=vmax)
    ax2d.contour(A, B, Z, levels=levels_arg, colors='k', linewidths=0.3, alpha=0.4)
    fig.colorbar(cf, ax=ax2d, label=Z_label, shrink=0.85)
    ax2d.set_xlabel(xlabel, fontsize=9)
    ax2d.set_ylabel(ylabel, fontsize=9)

    return cf


def _draw_trail(ax, trail_a: np.ndarray, trail_b: np.ndarray, color):
    """Draw a single colored trail with arrows on a 2D axes."""
    n = len(trail_a)
    if n < 2:
        return
    for i in range(n - 1):
        ax.plot([trail_a[i], trail_a[i + 1]], [trail_b[i], trail_b[i + 1]],
                color=color, linewidth=1.5, zorder=6, alpha=0.8)
        ax.annotate(
            '', xy=(trail_a[i + 1], trail_b[i + 1]),
            xytext=(trail_a[i], trail_b[i]),
            arrowprops=dict(arrowstyle='->', color=color, lw=1.5),
            zorder=7,
        )
    ax.scatter(trail_a, trail_b, color=color, s=14, zorder=8,
               edgecolors='none', alpha=0.7)


# ---------------------------------------------------------------------------
# Plot 1 — single-agent loss landscape (random filter-normalized directions)
# ---------------------------------------------------------------------------

def plot_loss_landscape(
    grid: dict,
    title: str = 'Loss Landscape',
    log_scale: bool = True,
    figsize: tuple[float, float] = (16, 6),
    elev: float = 30.0,
    azim: float = -60.0,
    vmin: float | None = None,
    vmax: float | None = None,
) -> plt.Figure:
    """
    Plot the loss landscape of a single agent as a 3D surface + 2D contour.

    Uses the random filter-normalized directions from compute_loss_grid.
    Rigorous and comparable across conditions — do not overlay other agents
    here since their projections onto random axes are meaningless.

    Parameters
    ----------
    grid : dict
        Output of compute_loss_grid().
    title : str
    log_scale : bool
        Plot log(loss) — makes basin edges more visible.
    figsize : tuple
    elev : float
    azim : float

    Returns
    -------
    matplotlib Figure
    """
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    loss_grid  = grid['loss_grid']
    alpha_vals = grid['alpha_vals']
    beta_vals  = grid['beta_vals']

    A, B    = np.meshgrid(alpha_vals, beta_vals, indexing='ij')
    Z       = np.log(loss_grid + 1e-8) if log_scale else loss_grid
    Z_label = 'log(loss)' if log_scale else 'loss'

    fig = plt.figure(figsize=figsize)
    fig.suptitle(title, fontsize=13, y=1.01)

    ax3d = fig.add_subplot(121, projection='3d')
    ax2d = fig.add_subplot(122)

    _draw_surface_and_contour(fig, ax3d, ax2d, A, B, Z, Z_label, elev, azim,
                               'α (direction δ)', 'β (direction η)',
                               vmin=vmin, vmax=vmax)

    ax3d.set_title('3D Surface', fontsize=11)
    ax2d.set_title('2D Contour', fontsize=11)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Plot 2 — multi-agent PCA plot
# ---------------------------------------------------------------------------

def plot_agent_pca(
    agents: list[AgentTrainer],
    criterion: nn.Module,
    loader: DataLoader,
    param_snapshots: list[list],
    agent_labels: list[str] | None = None,
    title: str = 'Agent PCA',
    log_scale: bool = True,
    grid_size: int = 21,
    padding: float = 0.3,
    figsize: tuple[float, float] = (16, 6),
    elev: float = 30.0,
    azim: float = -60.0,
    vmin: float | None = None,
    vmax: float | None = None,
    precomputed_grid: dict | None = None,
) -> plt.Figure:
    """
    Plot all agents' final positions and training trails on a loss surface
    computed along the top 2 PCA directions of the agent ensemble.

    Because the axes are derived from where the agents actually ended up,
    agent dots are always spread across the plot and trails are always
    visible — unlike the random-direction landscape where both collapse
    to a single point.

    Parameters
    ----------
    agents : list[AgentTrainer]
        All trained agents in the ensemble.
    criterion : nn.Module
    loader : DataLoader
        Probe loader for loss evaluation.
    param_snapshots : list of list of Tensor
        param_snapshots[epoch][agent_idx] = param vector (D,) on CPU.
        Collected during training via epoch_callback.
    agent_labels : list[str] or None
    title : str
    log_scale : bool
    grid_size : int
        Points per axis for the loss grid. 21 is fast; use 51 for final runs.
        Ignored when precomputed_grid is provided.
    padding : float
        Fraction of the agent spread to add around the grid boundary so
        no agent dot sits on the edge. Ignored when precomputed_grid is provided.
    figsize : tuple
    elev : float
    azim : float
    precomputed_grid : dict or None
        Output of compute_loss_grid() using PC1/PC2 axes. When provided the
        grid computation step is skipped entirely — useful in run_ablation
        where all grids are computed upfront to find the shared color scale,
        avoiding a redundant second pass over the data.

    Returns
    -------
    matplotlib Figure
    """
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    device = agents[0].device
    labels = agent_labels or [f'A{i}' for i in range(len(agents))]
    colors = plt.cm.tab10(np.linspace(0, 1, len(agents)))

    # ── 1. Stack final param vectors, compute mean and PCA ────────────────
    final_vecs = torch.stack(
        [a.param_vector().cpu() for a in agents]
    )                                               # (n_agents, D)
    mean_vec   = final_vecs.mean(dim=0)             # (D,)
    deviations = final_vecs - mean_vec              # (n_agents, D)

    # SVD of deviations → top 2 right singular vectors are PC1, PC2
    # deviations shape: (n_agents, D) — U (n×n), S (n,), Vt (n×D)
    _, S, Vt = torch.linalg.svd(deviations, full_matrices=False)
    pc1 = Vt[0]   # (D,)
    pc2 = Vt[1]   # (D,)

    # Variance explained by PC1+PC2
    var_total    = (S ** 2).sum().item()
    var_explained = ((S[0] ** 2 + S[1] ** 2) / var_total).item() if var_total > 0 else 1.0

    # ── 2. Project final positions onto (PC1, PC2) ────────────────────────
    final_a = (deviations @ pc1).numpy()   # (n_agents,)
    final_b = (deviations @ pc2).numpy()

    # ── 3. Auto-compute grid range to contain all agents + padding ────────
    #    Skipped when a pre-computed grid is provided (range already set).
    if precomputed_grid is None:
        spread_a = max(abs(final_a.max()), abs(final_a.min()))
        spread_b = max(abs(final_b.max()), abs(final_b.min()))
        spread   = max(spread_a, spread_b, 1e-3)
        r        = float(spread * (1.0 + padding))
        alpha_range = (-r, r)

    # ── 4. Compute loss grid using PC1/PC2 centered on mean ───────────────
    #    Reuse pre-computed grid when available to avoid redundant forward passes.
    if precomputed_grid is not None:
        grid = precomputed_grid
    else:
        grid = compute_loss_grid(
            agent       = agents[0],
            criterion   = criterion,
            loader      = loader,
            grid_size   = grid_size,
            alpha_range = alpha_range,
            center      = mean_vec,
            directions  = (pc1, pc2),
        )

    loss_grid  = grid['loss_grid']
    alpha_vals = grid['alpha_vals']
    beta_vals  = grid['beta_vals']

    A, B    = np.meshgrid(alpha_vals, beta_vals, indexing='ij')
    Z       = np.log(loss_grid + 1e-8) if log_scale else loss_grid
    Z_label = 'log(loss)' if log_scale else 'loss'

    # ── 5. Project trajectories onto (PC1, PC2) ───────────────────────────
    # trails[i] = (trail_a, trail_b) arrays for agent i
    trails = []
    for i in range(len(agents)):
        snaps = torch.stack(
            [param_snapshots[epoch][i] for epoch in range(len(param_snapshots))]
        )                                          # (n_epochs, D)
        dev_snaps = snaps - mean_vec.unsqueeze(0)
        t_a = (dev_snaps @ pc1).numpy()
        t_b = (dev_snaps @ pc2).numpy()
        trails.append((t_a, t_b))

    # ── 6. Build figure ───────────────────────────────────────────────────
    fig = plt.figure(figsize=figsize)
    fig.suptitle(
        f'{title}\nPC1+PC2 variance explained: {var_explained:.1%}',
        fontsize=13, y=1.01,
    )

    ax3d = fig.add_subplot(121, projection='3d')
    ax2d = fig.add_subplot(122)

    _draw_surface_and_contour(fig, ax3d, ax2d, A, B, Z, Z_label, elev, azim,
                               'PC1', 'PC2', vmin=vmin, vmax=vmax)

    # Interpolator for getting surface z at any (a, b)
    from scipy.interpolate import RegularGridInterpolator
    interp = RegularGridInterpolator(
        (alpha_vals, beta_vals), Z, method='linear', bounds_error=False, fill_value=None
    )

    # Mean position marker
    ax3d.plot3D([0], [0], [float(interp([[0, 0]]))], 'o',
                color='white', markersize=7,
                markeredgecolor='black', markeredgewidth=1.2,
                label='ensemble mean', zorder=5)
    ax2d.plot(0, 0, 'o', color='white', markersize=9,
              markeredgecolor='black', markeredgewidth=1.2,
              label='ensemble mean', zorder=5)

    # Agent final positions + trails
    for i, (label, color) in enumerate(zip(labels, colors)):
        a_i, b_i = float(final_a[i]), float(final_b[i])

        # 3D dot on the surface
        z_i = float(interp([[a_i, b_i]]))
        ax3d.plot3D([a_i], [b_i], [z_i], 'o',
                    color=color, markersize=7,
                    markeredgecolor='black', markeredgewidth=0.8,
                    label=label, zorder=7)

        # 2D trail
        _draw_trail(ax2d, trails[i][0], trails[i][1], color)

        # 2D final dot (on top of trail)
        ax2d.plot(a_i, b_i, 'o', color=color, markersize=8,
                  markeredgecolor='black', markeredgewidth=0.8,
                  label=label, zorder=9)

    ax3d.set_title('3D Surface', fontsize=11)
    ax3d.legend(fontsize=8, loc='upper left')

    ax2d.set_title('2D Contour', fontsize=11)
    ax2d.legend(fontsize=8, loc='upper left')

    fig.tight_layout()
    return fig
