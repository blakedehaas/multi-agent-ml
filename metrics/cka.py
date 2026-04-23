"""
metrics/cka.py

Centered Kernel Alignment (CKA) for measuring representational similarity
between agents at multiple layers and checkpoints.

Reference: Kornblith et al. 2019 — "Similarity of Neural Network Representations
Revisited" (https://arxiv.org/abs/1905.00414)

Core idea
---------
For a fixed probe dataset fed through two networks A and B:

    X  shape (N, D_a)  — activations from network A at some layer
    Y  shape (N, D_b)  — activations from network B at same layer

    K = X @ X.T        (N, N) Gram matrix — pairwise similarity in A's space
    L = Y @ Y.T        (N, N) Gram matrix — pairwise similarity in B's space

    Center both:  K_c = HKH,  L_c = HLH
    where H = I - (1/N) * 11^T  (centering matrix)

    HSIC(K, L) = (1/(N-1)²) * trace(K_c @ L_c)
    CKA(X, Y)  = HSIC(K, L) / sqrt(HSIC(K,K) * HSIC(L,L))

Result is in [0, 1]:
    1.0  →  identical representational structure
    0.0  →  completely unrelated representations

Key properties
--------------
- Invariant to orthogonal transformation and isotropic scaling of activations
- Works even when D_a ≠ D_b (comparison is always in N×N Gram space)
- Uses linear kernel (K = X @ X.T) — standard for neural network comparison

Usage
-----
    tracker = CKATracker(agents, probe_loader)

    # At each checkpoint during training:
    results = tracker.compute(epoch=5)

    # results['gap']['matrix']   →  (N_agents, N_agents) CKA matrix
    # results['gap']['mean_sim'] →  mean off-diagonal (diversity summary)
"""

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from baselines.single_trainer import AgentTrainer
from models.cnn import TinyNet


# ---------------------------------------------------------------------------
# Core CKA functions
# ---------------------------------------------------------------------------

@torch.no_grad()
def center_gram(K: Tensor) -> Tensor:
    """
    Apply double centering to a Gram matrix K.

    H @ K @ H  where  H = I - (1/N) * 11^T

    Computed without constructing H explicitly:
        K_c = K - row_means - col_means + grand_mean

    Parameters
    ----------
    K : Tensor shape (N, N)

    Returns
    -------
    Tensor shape (N, N) — centered Gram matrix
    """
    row_means   = K.mean(dim=1, keepdim=True)   # (N, 1)
    col_means   = K.mean(dim=0, keepdim=True)   # (1, N)
    grand_mean  = K.mean()                       # scalar
    return K - row_means - col_means + grand_mean


@torch.no_grad()
def linear_hsic(K_c: Tensor, L_c: Tensor) -> Tensor:
    """
    Compute HSIC between two centered Gram matrices.

    HSIC(K, L) = (1/(N-1)²) * trace(K_c @ L_c)

    Uses the identity trace(A @ B) = (A * B^T).sum() to avoid
    materializing the full N×N product matrix.
    Since K_c and L_c are symmetric, B^T = B, so:
        trace(K_c @ L_c) = (K_c * L_c).sum()

    Parameters
    ----------
    K_c : Tensor shape (N, N) — centered Gram matrix
    L_c : Tensor shape (N, N) — centered Gram matrix

    Returns
    -------
    Scalar tensor
    """
    N = K_c.shape[0]
    return (K_c * L_c).sum() / (N - 1) ** 2


@torch.no_grad()
def linear_cka(X: Tensor, Y: Tensor) -> float:
    """
    Compute linear CKA between two activation matrices.

    Parameters
    ----------
    X : Tensor shape (N, D_a) — activations from network A
    Y : Tensor shape (N, D_b) — activations from network B
        N must match; D_a and D_b may differ.

    Returns
    -------
    float in [0, 1]
        1.0 = identical representational structure
        0.0 = completely unrelated representations
    """
    if X.shape[0] != Y.shape[0]:
        raise ValueError(
            f"X and Y must have the same number of samples. "
            f"Got X: {X.shape[0]}, Y: {Y.shape[0]}"
        )

    # Move to float32 — avoid precision issues with half-precision activations
    X = X.float()
    Y = Y.float()

    # Gram matrices
    K = X @ X.T   # (N, N)
    L = Y @ Y.T   # (N, N)

    # Center
    K_c = center_gram(K)
    L_c = center_gram(L)

    # HSIC values
    hsic_kl = linear_hsic(K_c, L_c)
    hsic_kk = linear_hsic(K_c, K_c)
    hsic_ll = linear_hsic(L_c, L_c)

    # Normalize — guard against degenerate zero activations
    denom = (hsic_kk * hsic_ll).clamp(min=1e-10).sqrt()
    cka   = (hsic_kl / denom).clamp(min=0.0, max=1.0)

    return cka.item()


# ---------------------------------------------------------------------------
# Activation extraction
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_probe_activations(
    agent: AgentTrainer,
    probe_loader: DataLoader,
) -> dict[str, Tensor]:
    """
    Run the probe dataset through one agent and collect activations at all
    CKA probe points.

    Uses TinyNet.forward_with_probes() to collect named intermediate
    representations in a single forward pass.

    Parameters
    ----------
    agent : AgentTrainer
    probe_loader : DataLoader — fixed probe set from get_probe_loader()

    Returns
    -------
    dict mapping layer name -> Tensor shape (N_probe, D_layer)
        Keys: 'block1', 'block2', 'block3', 'gap'
    """
    # Unwrap torch.compile's OptimizedModule to access TinyNet directly.
    # _orig_mod holds the original model when torch.compile has been applied.
    raw_model = getattr(agent.model, '_orig_mod', agent.model)

    if not isinstance(raw_model, TinyNet):
        raise TypeError(
            f"CKA extraction requires a TinyNet model, got {type(raw_model).__name__}. "
            f"Ensure agents are built with TinyNet."
        )

    agent.model.eval()

    all_probes: dict[str, list[Tensor]] = {}

    for X, _ in probe_loader:
        X = X.to(agent.device)
        _, batch_probes = raw_model.forward_with_probes(X)
        for layer_name, acts in batch_probes.items():
            all_probes.setdefault(layer_name, []).append(acts.cpu())

    # Concatenate batches → (N_probe, D_layer) per layer
    return {name: torch.cat(batches, dim=0) for name, batches in all_probes.items()}


# ---------------------------------------------------------------------------
# CKA Tracker
# ---------------------------------------------------------------------------

class CKATracker:
    """
    Compute and store pairwise CKA matrices across all agents and layers
    at training checkpoints.

    Parameters
    ----------
    agents : list[AgentTrainer]
        The ensemble being trained.
    probe_loader : DataLoader
        Fixed probe set from get_probe_loader(). Must never be shuffled.
    layers : list[str]
        Which TinyNet probe points to compute CKA at.
        Default: all four ('block1', 'block2', 'block3', 'gap').

    Usage
    -----
        tracker = CKATracker(agents, probe_loader)

        # inside training loop, every eval_interval epochs:
        results = tracker.compute(epoch=current_epoch)

        # Log to wandb:
        for layer, stats in results.items():
            wandb.log({
                f'cka/{layer}/mean_sim': stats['mean_sim'],
                f'cka/{layer}/matrix':  wandb.Image(plot_cka_matrix(stats['matrix']))
            }, step=current_epoch)
    """

    LAYER_NAMES = ('block1', 'block2', 'block3', 'gap')

    def __init__(
        self,
        agents: list[AgentTrainer],
        probe_loader: DataLoader,
        layers: list[str] | None = None,
    ) -> None:
        self.agents       = agents
        self.probe_loader = probe_loader
        self.layers       = list(layers) if layers else list(self.LAYER_NAMES)
        self.history: list[dict] = []   # stores results from each compute() call

    def compute(self, epoch: int) -> dict[str, dict]:
        """
        Extract probe activations from all agents and compute the pairwise
        CKA matrix at each tracked layer.

        Parameters
        ----------
        epoch : int
            Current training epoch — stored in history for later plotting.

        Returns
        -------
        dict mapping layer_name -> {
            'matrix'   : Tensor (N_agents, N_agents) — pairwise CKA scores
            'mean_sim' : float  — mean off-diagonal value (diversity summary)
            'min_sim'  : float  — min off-diagonal (most diverse pair)
            'max_sim'  : float  — max off-diagonal (most similar pair)
        }
        """
        n = len(self.agents)

        # Step 1: extract activations for all agents (one forward pass each)
        all_activations: list[dict[str, Tensor]] = [
            extract_probe_activations(agent, self.probe_loader)
            for agent in self.agents
        ]

        results: dict[str, dict] = {}

        for layer in self.layers:
            # Step 2: build (N, N) CKA matrix for this layer
            cka_matrix = torch.zeros(n, n)

            for i in range(n):
                cka_matrix[i, i] = 1.0   # self-similarity is always 1
                for j in range(i + 1, n):
                    score = linear_cka(
                        all_activations[i][layer],
                        all_activations[j][layer],
                    )
                    cka_matrix[i, j] = score
                    cka_matrix[j, i] = score   # symmetric

            # Step 3: compute off-diagonal summary statistics
            mask        = ~torch.eye(n, dtype=torch.bool)
            off_diag    = cka_matrix[mask]

            results[layer] = {
                'matrix':   cka_matrix,
                'mean_sim': off_diag.mean().item(),
                'min_sim':  off_diag.min().item(),
                'max_sim':  off_diag.max().item(),
            }

        # Store in history
        self.history.append({'epoch': epoch, **results})

        return results

    def mean_similarity_over_time(self, layer: str) -> tuple[list[int], list[float]]:
        """
        Return (epochs, mean_sim_values) for a given layer across all
        recorded checkpoints. Useful for plotting diversity curves.
        """
        epochs   = [entry['epoch'] for entry in self.history]
        mean_sim = [entry[layer]['mean_sim'] for entry in self.history]
        return epochs, mean_sim

    def __repr__(self) -> str:
        return (
            f"CKATracker("
            f"n_agents={len(self.agents)}, "
            f"layers={self.layers}, "
            f"checkpoints={len(self.history)})"
        )
