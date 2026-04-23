"""
swarm/trainer.py

SwarmTrainer — the full Boids-style ensemble trainer.

Extends EnsembleTrainer by overriding the two hooks to inject the three
swarm interaction rules at the correct points in the training loop:

    ┌─────────────────────────────────────────────────────────────┐
    │  for each batch:                                            │
    │                                                             │
    │    1. forward + backward  (all agents, no optimizer step)   │
    │                                                             │
    │    2. pre_gradient_step()          ← HOOK (alignment)       │
    │         · build/refresh k-NN graph from current params      │
    │         · blend each agent's gradient with neighbor mean    │
    │         · write blended gradients back before step()        │
    │                                                             │
    │    3. optimizer.step()  (all agents)                        │
    │                                                             │
    │    4. post_gradient_step()         ← HOOK (sep + coh)       │
    │         · read updated param vectors                        │
    │         · compute separation + cohesion deltas              │
    │         · write corrected param vectors back                │
    └─────────────────────────────────────────────────────────────┘

Warm-start gating (from TrainingConfig.warm_start_steps) is handled by
the base class — both hooks are skipped until that many steps have passed.

SwarmConfig controls which rules are active:
    alpha = 0  →  alignment  disabled
    beta  = 0  →  separation disabled
    gamma = 0  →  cohesion   disabled

Setting all three to zero reproduces IndependentEnsembleTrainer exactly.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn

from baselines.single_trainer import AgentTrainer, EnsembleTrainer
from swarm.topology import KNNTopology
from swarm.rules import apply_rules


# ---------------------------------------------------------------------------
# Swarm-specific configuration
# ---------------------------------------------------------------------------

@dataclass
class SwarmConfig:
    """
    Hyperparameters that control the swarm interaction rules.

    Kept separate from TrainingConfig so you can mix-and-match
    optimizer settings and swarm settings independently.

    Attributes
    ----------
    alpha : float in [0, 1]
        Gradient alignment strength.
        0 = no alignment (rule disabled).
        1 = agent fully adopts neighborhood mean gradient.

    beta : float >= 0
        Separation strength. Controls the magnitude of the repulsive force
        pushing agents apart in parameter space. Larger β → wider spacing.

    gamma : float >= 0
        Cohesion strength. Controls the magnitude of the constant-force
        pull toward the neighborhood centroid. Larger γ → tighter spacing.

        Equilibrium spacing: dist* = β / γ  (when both rules are active).

    k : int >= 1
        Number of nearest neighbors each agent interacts with.
        Set k = N - 1 for full all-to-all connectivity.

    update_interval : int >= 1
        How many training steps between k-NN graph recomputations.
        1 = recompute every step (most accurate, most expensive).
        Larger values reduce overhead but the graph lags agent movement.

    eps : float
        Numerical stability constant shared by separation and cohesion.
    """
    alpha: float = 0.0
    beta: float  = 0.0
    gamma: float = 0.0
    k: int = 3
    update_interval: int = 1
    eps: float = 1e-8

    def __post_init__(self) -> None:
        if not (0.0 <= self.alpha <= 1.0):
            raise ValueError(f"alpha must be in [0, 1], got {self.alpha}")
        if self.beta < 0:
            raise ValueError(f"beta must be >= 0, got {self.beta}")
        if self.gamma < 0:
            raise ValueError(f"gamma must be >= 0, got {self.gamma}")
        if self.k < 1:
            raise ValueError(f"k must be >= 1, got {self.k}")
        if self.update_interval < 1:
            raise ValueError(f"update_interval must be >= 1, got {self.update_interval}")

    @property
    def any_rule_active(self) -> bool:
        """True if at least one swarm rule has a non-zero coefficient."""
        return self.alpha > 0 or self.beta > 0 or self.gamma > 0

    @property
    def label(self) -> str:
        """Short human-readable tag for wandb run naming."""
        return f"a{self.alpha}_b{self.beta}_g{self.gamma}_k{self.k}"


# ---------------------------------------------------------------------------
# SwarmTrainer
# ---------------------------------------------------------------------------

class SwarmTrainer(EnsembleTrainer):
    """
    Ensemble trainer with Boids-style swarm interaction rules.

    Parameters
    ----------
    agents : list[AgentTrainer]
        The ensemble. All agents must share the same architecture and device.
    criterion : nn.Module
        Loss function (e.g. nn.CrossEntropyLoss()).
    swarm_cfg : SwarmConfig
        Controls which rules are active and at what strength.

    Notes
    -----
    - The k-NN graph is built in pre_gradient_step() and the SAME cached
      graph is reused in post_gradient_step() within the same training step.
      This ensures the alignment and position corrections are consistent —
      agents interact with the same set of neighbors throughout one step.

    - If swarm_cfg.any_rule_active is False, this trainer is functionally
      identical to IndependentEnsembleTrainer (both hooks become no-ops via
      the fast-paths in rules.py).
    """

    def __init__(
        self,
        agents: list[AgentTrainer],
        criterion: nn.Module,
        swarm_cfg: SwarmConfig,
    ) -> None:
        super().__init__(agents, criterion)
        self.swarm_cfg = swarm_cfg
        self.topology  = KNNTopology(
            k=swarm_cfg.k,
            update_interval=swarm_cfg.update_interval,
        )

    # ------------------------------------------------------------------
    # Hook 1 — gradient alignment (pre optimizer.step)
    # ------------------------------------------------------------------

    def pre_gradient_step(self) -> None:
        """
        Blend each agent's gradient with the mean gradient of its k neighbors.

        Steps
        -----
        1. Collect current param vectors → build/refresh k-NN graph.
           (Topology is based on where agents ARE, before this step moves them.)
        2. Collect current grad vectors.
        3. Call gradient_alignment via apply_rules (returns new grad vectors).
        4. Write new gradients back into each agent before optimizer.step().

        If alpha = 0 the apply_rules fast-path returns the original gradients
        unchanged, so this hook costs only the param_vector() reads and the
        topology check.
        """
        if self.swarm_cfg.alpha == 0.0:
            return  # fast-path: nothing to do

        # Step 1: build/refresh topology from current parameter positions
        param_vecs  = [a.param_vector() for a in self.agents]
        neighbor_map = self.topology.get_neighbors(param_vecs, step=self._global_step)

        # Step 2: collect gradients (zero-fill agents with no grad)
        grad_vecs = []
        for a in self.agents:
            g = a.grad_vector()
            grad_vecs.append(g if g is not None else torch.zeros_like(param_vecs[0]))

        # Step 3 & 4: blend and write back
        new_grads, _ = apply_rules(
            grad_vectors  = grad_vecs,
            param_vectors = param_vecs,
            neighbor_map  = neighbor_map,
            alpha         = self.swarm_cfg.alpha,
            beta          = 0.0,   # position rules run post-step
            gamma         = 0.0,
            eps           = self.swarm_cfg.eps,
        )

        for agent, new_g in zip(self.agents, new_grads):
            agent.set_grad_vector(new_g)

    # ------------------------------------------------------------------
    # Hook 2 — separation + cohesion (post optimizer.step)
    # ------------------------------------------------------------------

    def post_gradient_step(self) -> None:
        """
        Apply separation and cohesion forces to updated parameter vectors.

        Steps
        -----
        1. Collect UPDATED param vectors (post optimizer.step).
        2. Retrieve the neighbor map cached in pre_gradient_step — uses the
           same topology from this step for consistency.
           If only position rules are active (alpha=0), the cache may not
           exist yet, so get_neighbors is called here instead.
        3. Compute sep + coh deltas via apply_rules.
        4. Add deltas to each agent's current parameters.

        If beta = gamma = 0 this hook is effectively a no-op.
        """
        if self.swarm_cfg.beta == 0.0 and self.swarm_cfg.gamma == 0.0:
            return  # fast-path: nothing to do

        # Step 1: read updated params
        param_vecs   = [a.param_vector() for a in self.agents]

        # Step 2: get neighbor map (may be cache from pre_gradient_step)
        neighbor_map = self.topology.get_neighbors(param_vecs, step=self._global_step)

        # Step 3: compute position deltas (grad rules are no-ops here)
        _, param_deltas = apply_rules(
            grad_vectors  = [torch.zeros_like(p) for p in param_vecs],  # unused
            param_vectors = param_vecs,
            neighbor_map  = neighbor_map,
            alpha         = 0.0,   # gradient rule ran pre-step
            beta          = self.swarm_cfg.beta,
            gamma         = self.swarm_cfg.gamma,
            eps           = self.swarm_cfg.eps,
        )

        # Step 4: write corrected parameters back
        for agent, p_vec, delta in zip(self.agents, param_vecs, param_deltas):
            agent.set_param_vector(p_vec + delta)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    @torch.no_grad()
    def inter_agent_distances(self) -> torch.Tensor:
        """
        Return the (N, N) pairwise L2 distance matrix in parameter space.
        Convenience wrapper around KNNTopology.pairwise_distances().
        Useful for logging diversity during training.

        Always returns a CPU tensor — safe to pass directly to plotting
        and logging functions regardless of training device.
        """
        param_vecs = [a.param_vector() for a in self.agents]
        return self.topology.pairwise_distances(param_vecs).cpu()

    def __repr__(self) -> str:
        return (
            f"SwarmTrainer("
            f"n_agents={len(self.agents)}, "
            f"α={self.swarm_cfg.alpha}, "
            f"β={self.swarm_cfg.beta}, "
            f"γ={self.swarm_cfg.gamma}, "
            f"k={self.swarm_cfg.k})"
        )
