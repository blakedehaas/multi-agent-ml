"""
swarm/topology.py

Dynamic k-NN connectivity graph over agent parameter space.

Each agent has exactly k neighbors — the k agents currently closest to it
in parameter space (L2 distance over flattened weight vectors). The graph
is directed: if agent A lists agent B as a neighbor, B does not necessarily
list A back.

Setting k = N - 1 (where N is the total number of agents) recovers full
all-to-all connectivity as a special case.

The graph is recomputed every `update_interval` training steps so it tracks
agents as they move through parameter space during training. Recomputing
every step is the most accurate but also most expensive; every epoch is
cheap but can be stale in early training where agents move fast.

Usage
-----
    topology = KNNTopology(k=3, update_interval=10)

    # Inside training loop, after collecting param vectors:
    param_vecs = [agent.param_vector() for agent in agents]
    neighbor_map = topology.get_neighbors(param_vecs, step=current_step)

    # neighbor_map[i] -> list of k agent indices closest to agent i
    for i, neighbors in neighbor_map.items():
        print(f"Agent {i} neighbors: {neighbors}")
"""

import torch
from typing import Dict, List, Optional


class KNNTopology:
    """
    Dynamic k-nearest-neighbor graph over agent parameter space.

    Parameters
    ----------
    k : int
        Number of neighbors per agent. Must satisfy 1 <= k <= N-1 where N
        is the number of agents. Use k = N-1 for full connectivity.
    update_interval : int
        Recompute the graph every this many training steps.
        Default 1 recomputes every step (most accurate).
        Increase to reduce overhead in large swarms.

    Attributes
    ----------
    k : int
    update_interval : int
    _cache : dict or None
        Most recently computed neighbor map, keyed by agent index.
    _last_update : int
        Training step at which the graph was last recomputed.
    """

    def __init__(self, k: int, update_interval: int = 1) -> None:
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        self.k = k
        self.update_interval = update_interval
        self._cache: Optional[Dict[int, List[int]]] = None
        self._last_update: int = -1

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def get_neighbors(
        self,
        param_vectors: List[torch.Tensor],
        step: int = 0,
    ) -> Dict[int, List[int]]:
        """
        Return the neighbor map for the current step.

        The graph is recomputed from scratch if:
          - It has never been computed, OR
          - (step - last_update) >= update_interval

        Otherwise the cached graph is returned unchanged.

        Parameters
        ----------
        param_vectors : list of Tensor, each shape (D,)
            Flattened parameter vector for each agent, in agent-index order.
            Obtain via AgentTrainer.param_vector().
        step : int
            Current global training step. Used to decide whether to refresh.

        Returns
        -------
        dict mapping agent index (int) -> list of k neighbor indices (List[int])
            Neighbors are sorted nearest-first.
        """
        n_agents = len(param_vectors)

        # Validate k against current swarm size
        if self.k >= n_agents:
            raise ValueError(
                f"k={self.k} must be < number of agents ({n_agents}). "
                f"For full connectivity use k={n_agents - 1}."
            )

        # Return cache if still fresh
        if self._cache is not None and (step - self._last_update) < self.update_interval:
            return self._cache

        self._cache = self._compute(param_vectors)
        self._last_update = step
        return self._cache

    def force_recompute(self, param_vectors: List[torch.Tensor], step: int = 0) -> Dict[int, List[int]]:
        """
        Bypass the cache and recompute the graph immediately.

        Useful at the start of each epoch if update_interval > 1 but you
        still want a fresh graph at epoch boundaries.
        """
        self._cache = self._compute(param_vectors)
        self._last_update = step
        return self._cache

    @property
    def last_update_step(self) -> int:
        """Step at which the graph was last computed (-1 if never)."""
        return self._last_update

    # ------------------------------------------------------------------
    # Internal: pairwise distance + k-NN selection
    # ------------------------------------------------------------------
    def _compute(self, param_vectors: List[torch.Tensor]) -> Dict[int, List[int]]:
        """
        Build the k-NN graph from current parameter vectors.

        Steps
        -----
        1. Stack all param vectors into an (N, D) matrix.
        2. Compute the (N, N) pairwise squared-L2 distance matrix.
        3. For each agent i, find the k smallest distances (excluding self).
        4. Return as a dict.

        Complexity: O(N^2 * D) time, O(N^2) extra space.
        For N=10 agents this is negligible.
        """
        # Stack -> (N, D)
        P = torch.stack(param_vectors)          # (N, D)
        n = P.shape[0]

        # Pairwise squared L2: ||p_i - p_j||^2
        # = ||p_i||^2 + ||p_j||^2 - 2 * p_i . p_j
        sq_norms = (P * P).sum(dim=1, keepdim=True)  # (N, 1)
        dist_sq  = sq_norms + sq_norms.T - 2.0 * (P @ P.T)  # (N, N)

        # Numerical guard: distances should be >= 0
        dist_sq = dist_sq.clamp(min=0.0)

        # Mask self-distance so agent i is never its own neighbor
        dist_sq.fill_diagonal_(float('inf'))

        # For each agent, find k nearest
        _, idx = torch.topk(dist_sq, k=self.k, dim=1, largest=False, sorted=True)  # (N, k)

        neighbor_map: Dict[int, List[int]] = {
            i: idx[i].tolist() for i in range(n)
        }
        return neighbor_map

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    def pairwise_distances(self, param_vectors: List[torch.Tensor]) -> torch.Tensor:
        """
        Return the full (N, N) L2 distance matrix (not squared).

        Useful for logging average inter-agent distance as a diversity metric.

        Returns
        -------
        Tensor of shape (N, N), dtype float32.
        Diagonal is 0.
        """
        P = torch.stack(param_vectors)
        sq_norms = (P * P).sum(dim=1, keepdim=True)
        dist_sq  = (sq_norms + sq_norms.T - 2.0 * (P @ P.T)).clamp(min=0.0)
        dist_sq.fill_diagonal_(0.0)
        return dist_sq.sqrt()

    def __repr__(self) -> str:
        return (
            f"KNNTopology(k={self.k}, update_interval={self.update_interval}, "
            f"last_update={self._last_update})"
        )
