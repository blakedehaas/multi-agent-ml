"""
swarm/rules.py

The three swarm interaction rules that form the ablation matrix.

Each rule is a pure function — it takes current state (gradients or parameters)
plus the neighbor map from topology.py, and returns *delta tensors* to be
applied by the trainer. Nothing is modified in-place here.

                    ┌──────────────────────────────────────┐
                    │          Ablation Matrix             │
                    │  α (alignment) ─ gradient level      │
                    │  β (separation) ─ parameter level    │
                    │  γ (cohesion)   ─ parameter level    │
                    │                                      │
                    │  Baseline: α=0, β=0, γ=0             │
                    └──────────────────────────────────────┘

Timing in the training loop
---------------------------
    # 1. Forward + backward for all agents
    loss.backward()

    # 2. PRE-STEP hook  ← gradient_alignment lives here
    new_grads = gradient_alignment(grad_vecs, neighbor_map, alpha)
    agent.set_grad_vector(new_grads[i])

    # 3. optimizer.step() for all agents

    # 4. POST-STEP hook  ← separation + cohesion live here
    sep_deltas = separation(param_vecs, neighbor_map, beta)
    coh_deltas = cohesion(param_vecs, neighbor_map, gamma)
    agent.set_param_vector(param_vecs[i] + sep_deltas[i] + coh_deltas[i])
"""

import torch
from torch import Tensor
from typing import Dict, List


# ---------------------------------------------------------------------------
# Rule 1 — Gradient Alignment  (α)
# ---------------------------------------------------------------------------

def gradient_alignment(
    grad_vectors: List[Tensor],
    neighbor_map: Dict[int, List[int]],
    alpha: float,
) -> List[Tensor]:
    """
    Blend each agent's gradient with the mean gradient of its k neighbors.

    Formula
    -------
        g_i_new = (1 - α) * g_i  +  α * mean(g_j  for j in neighbors(i))

    Intuition
    ---------
    α = 0  →  pure independent gradient descent (baseline)
    α = 1  →  agent ignores its own gradient and follows neighborhood average
    α ∈ (0,1) →  partial consensus: agents nudge each other toward agreement
                  on gradient direction, smoothing out conflicting updates

    This operates BEFORE optimizer.step() so the blended gradient is what
    the optimizer actually uses to update parameters.

    Parameters
    ----------
    grad_vectors : list of Tensor, each shape (D,)
        Flattened gradient vectors, one per agent.
        Agents with None gradients (e.g. frozen layers) should pass zeros.
    neighbor_map : dict
        {agent_idx: [neighbor_idx, ...]} from KNNTopology.get_neighbors()
    alpha : float in [0, 1]
        Alignment strength. 0 = no alignment (baseline).

    Returns
    -------
    list of Tensor, same shapes as input — new gradient vectors.
    """
    if alpha == 0.0:
        return list(grad_vectors)  # fast-path: no-op for baseline

    new_grads = []
    for i, g_i in enumerate(grad_vectors):
        neighbor_indices = neighbor_map[i]
        neighbor_grads = torch.stack([grad_vectors[j] for j in neighbor_indices])
        g_mean = neighbor_grads.mean(dim=0)                   # mean neighbor gradient
        g_new  = (1.0 - alpha) * g_i + alpha * g_mean
        new_grads.append(g_new)

    return new_grads


# ---------------------------------------------------------------------------
# Rule 2 — Separation  (β)
# ---------------------------------------------------------------------------

def separation(
    param_vectors: List[Tensor],
    neighbor_map: Dict[int, List[int]],
    beta: float,
    eps: float = 1e-8,
) -> List[Tensor]:
    """
    Apply a repulsive force pushing each agent away from its neighbors.

    Formula
    -------
        For each neighbor j of agent i:
            direction_ij = (θ_i - θ_j) / (||θ_i - θ_j||^2 + ε)

        Δθ_i = β * Σ_j  direction_ij

    Intuition
    ---------
    β = 0  →  no separation (baseline)
    β > 0  →  agents are pushed apart in parameter space, promoting diversity.
              The force is STRONGER the CLOSER two agents are (inverse-square
              weighting), so it mainly prevents collapse rather than splitting
              agents that are already distant.

    This operates AFTER optimizer.step(). The deltas are added to the
    updated parameter vectors.

    Parameters
    ----------
    param_vectors : list of Tensor, each shape (D,)
    neighbor_map : dict
    beta : float >= 0
        Separation strength. 0 = no separation (baseline).
    eps : float
        Numerical stability constant. Prevents division by zero when two
        agents occupy nearly the same point in parameter space.

    Returns
    -------
    list of Tensor — delta vectors (Δθ_i) to ADD to each agent's parameters.
    Zeros when beta = 0.
    """
    if beta == 0.0:
        return [torch.zeros_like(p) for p in param_vectors]  # fast-path

    deltas = []
    for i, theta_i in enumerate(param_vectors):
        force = torch.zeros_like(theta_i)
        for j in neighbor_map[i]:
            diff = theta_i - param_vectors[j]               # direction away from j
            dist_sq = (diff * diff).sum()                    # ||θ_i - θ_j||^2
            force += diff / (dist_sq + eps)                  # inverse-square weighting
        deltas.append(beta * force)

    return deltas


# ---------------------------------------------------------------------------
# Rule 3 — Cohesion  (γ)
# ---------------------------------------------------------------------------

def cohesion(
    param_vectors: List[Tensor],
    neighbor_map: Dict[int, List[int]],
    gamma: float,
    eps: float = 1e-8,
) -> List[Tensor]:
    """
    Pull each agent toward the centroid of its neighborhood with constant
    force magnitude (normalized cohesion).

    Formula
    -------
        centroid_i = mean(θ_j  for j in neighbors(i))
        direction  = centroid_i - θ_i
        Δθ_i = γ * direction / (||direction|| + ε)

    Why normalized?
    ---------------
    The linear spring formulation (Δθ_i = γ * (centroid - θ_i)) has force
    proportional to distance from the centroid. This means agents that have
    drifted far get yanked back hard — exactly the wrong behavior when we
    want separation to keep them spread out.

    Normalizing gives a constant-magnitude pull of γ steps toward the
    centroid regardless of how far away it is. Combined with inverse-square
    separation, this creates a stable equilibrium spacing:

        Equilibrium:  β / dist*²  =  γ   →   dist* = sqrt(β / γ)

    The ratio β/γ is therefore the key design knob controlling how far
    apart agents settle in parameter space:

        dist* = β / γ

    Note: even though separation uses an inverse-*square* denominator on the
    vector, the force magnitude scales as β/dist (inverse-linear) because
    the numerator already contains dist (||diff|| = dist). The squared term
    in the denominator cancels one factor of dist from the numerator.

    γ = 0       →  no cohesion (baseline)
    γ small     →  gentle drift inward, equilibrium spacing is large
    γ large     →  strong pull inward, equilibrium spacing is small
    no upper bound risk — force magnitude is always exactly γ

    This operates AFTER optimizer.step(). Combine with separation deltas:
        θ_i ← θ_i  +  Δθ_sep  +  Δθ_coh

    Parameters
    ----------
    param_vectors : list of Tensor, each shape (D,)
    neighbor_map : dict
    gamma : float >= 0
        Cohesion strength. 0 = no cohesion (baseline).
    eps : float
        Numerical stability constant. Prevents division by zero when agent
        is already exactly at the centroid.

    Returns
    -------
    list of Tensor — delta vectors (Δθ_i) to ADD to each agent's parameters.
    Zeros when gamma = 0 or agent is already at the centroid.
    """
    if gamma == 0.0:
        return [torch.zeros_like(p) for p in param_vectors]  # fast-path

    deltas = []
    for i, theta_i in enumerate(param_vectors):
        neighbor_params = torch.stack([param_vectors[j] for j in neighbor_map[i]])
        centroid  = neighbor_params.mean(dim=0)              # neighborhood centroid
        direction = centroid - theta_i                       # vector toward centroid
        dist      = direction.norm()                         # how far away centroid is
        deltas.append(gamma * direction / (dist + eps))      # normalized: constant magnitude

    return deltas


# ---------------------------------------------------------------------------
# Convenience: apply all three rules in one call
# ---------------------------------------------------------------------------

def apply_rules(
    grad_vectors: List[Tensor],
    param_vectors: List[Tensor],
    neighbor_map: Dict[int, List[int]],
    alpha: float,
    beta: float,
    gamma: float,
    eps: float = 1e-8,
) -> tuple[List[Tensor], List[Tensor]]:
    """
    Apply all three rules and return (new_grads, param_deltas).

    new_grads    — use these in place of grad_vectors before optimizer.step()
    param_deltas — add these to param_vectors after optimizer.step()

    Separation and cohesion deltas are summed into a single param_delta per
    agent so only one set_param_vector() call is needed.

    Parameters
    ----------
    grad_vectors  : list of Tensor (D,) — current gradients, pre-step
    param_vectors : list of Tensor (D,) — current parameters, pre-step
    neighbor_map  : dict from KNNTopology
    alpha, beta, gamma : rule strengths (0 = rule disabled)
    eps : stability constant for separation

    Returns
    -------
    new_grads    : list of Tensor (D,)
    param_deltas : list of Tensor (D,)   (sep + coh combined)
    """
    new_grads    = gradient_alignment(grad_vectors, neighbor_map, alpha)
    sep_deltas   = separation(param_vectors, neighbor_map, beta, eps)
    coh_deltas   = cohesion(param_vectors, neighbor_map, gamma)
    param_deltas = [s + c for s, c in zip(sep_deltas, coh_deltas)]

    return new_grads, param_deltas
