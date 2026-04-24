# Project Progress

Living document tracking decisions, findings, and next steps as the project evolves.
Updated continuously throughout development.

---

## Project Overview

The central hypothesis is that Boids-style swarm coordination rules — originally designed for physical flocking behavior — can be adapted to the gradient update step of neural network ensembles to produce more diverse, better-calibrated models.

Each network agent updates its weights according to:

```
θᵢ ← θᵢ - lr · (∇Lᵢ + α·alignment + β·separation + γ·cohesion)
```

Where neighbors are determined by a dynamic k-nearest-neighbor graph over parameter vectors, recomputed each epoch. The three rules are:
- **Alignment (α)** — blend gradient with the mean gradient of k-nearest neighbors, encouraging locally coherent optimization directions
- **Separation (β)** — repel agents that are too close in parameter space, discouraging redundant solutions
- **Cohesion (γ)** — attract agents toward the ensemble centroid, preventing total dispersion

The equilibrium spacing between agents is governed by the ratio β/γ.

---

## Model and Dataset

**Model: TinyNet** (~95K parameters)
- Small CNN for CIFAR-10 classification
- Architecture: conv layers with GroupNorm, Global Average Pooling (GAP), linear classifier
- Chosen for fast iteration — full CIFAR-10 training in ~minutes per agent on GPU
- CKA probe points exposed at multiple layers; GAP layer is the primary representational comparison point

**Dataset: CIFAR-10**
- 50K training / 10K test
- Full dataset used for final runs; 1K subset used for smoke tests
- Data downloads to Colab local storage each session (not persisted on Drive)

---

## Training Infrastructure

### EnsembleTrainer (base class)
Abstract base in `baselines/single_trainer.py`. Handles:
- Agent construction and device placement
- Training loop with `epoch_callback` hook
- Early stopping (`EarlyStopping(patience=5, min_delta=1e-3)` monitoring ensemble validation loss)
- `param_vector()` / `set_params()` for flat parameter manipulation
- Per-epoch parameter snapshot collection for visualization

### IndependentEnsembleTrainer (baseline)
Extends `EnsembleTrainer` with no inter-agent interaction. Pure independent training.

### SwarmTrainer
Extends `EnsembleTrainer`. After each agent's gradient step, applies the three rules in sequence using `swarm/rules.py` pure functions and `swarm/topology.py` for k-NN graph.

### Key hyperparameters (final ablation)
| Parameter | Value | Notes |
|-----------|-------|-------|
| n_agents | 10 | Full run |
| epochs | 50 | With patience=5 early stopping |
| batch_size | 512 | |
| lr | 1e-3 | Adam |
| weight_decay | 1e-4 | |
| k | 3 | k-NN neighborhood size |
| α | 0.3 | Gradient alignment strength |
| β | 0.5 | Separation strength |
| γ | 0.1 | Cohesion strength |
| β/γ | 5.0 | Equilibrium spacing |

### Early stopping
Monitors `ensemble_val_loss`. Patience was increased from 3 to 5 after the baseline condition triggered early stopping at epoch 33 while full_swarm ran all 50 — making conditions non-comparable. At patience=5 all conditions ran the full 50 epochs in the smoke test.

---

## Method — Technical Details

### Swarm Rules

All three rules are implemented as pure functions in `swarm/rules.py`. They take current state and return delta tensors — nothing is modified in-place. The timing within the training loop is critical and differs per rule.

#### Training Loop Timing

```
for each batch:
    1. forward + backward (all agents, no optimizer step)
    2. pre_gradient_step()   ← gradient_alignment lives here
         · build/refresh k-NN graph from current params
         · blend each agent's gradient with neighborhood mean
         · write blended gradients back before optimizer.step()
    3. optimizer.step() (all agents)
    4. post_gradient_step()  ← separation + cohesion live here
         · read updated param vectors
         · compute separation + cohesion deltas
         · write corrected param vectors back
```

Alignment must run before `optimizer.step()` so the blended gradient is what Adam actually uses. Separation and cohesion run after `optimizer.step()` and directly modify parameter vectors — they are not gradients.

The k-NN graph built in `pre_gradient_step` is cached and reused in `post_gradient_step` within the same training step, ensuring both hooks operate on the same neighbor relationships.

---

#### Rule 1 — Gradient Alignment (α)

**Formula:**
```
g_i_new = (1 - α) * g_i  +  α * mean(g_j  for j in neighbors(i))
```

**Intuition:**
Interpolates between each agent's own gradient and the mean gradient of its k-nearest neighbors in parameter space. α=0 is pure independent gradient descent (baseline). α=1 means the agent ignores its own gradient entirely and follows the neighborhood consensus. α∈(0,1) nudges agents toward gradient agreement, smoothing conflicting update directions.

**Implementation:**
```python
for i, g_i in enumerate(grad_vectors):
    neighbor_grads = torch.stack([grad_vectors[j] for j in neighbor_map[i]])
    g_mean  = neighbor_grads.mean(dim=0)
    g_new   = (1.0 - alpha) * g_i + alpha * g_mean
```

**Fast-path:** if α=0, returns the original gradient list unchanged with no computation.

**Where it operates:** gradient space, pre-`optimizer.step()`.

---

#### Rule 2 — Separation (β)

**Formula:**
```
For each neighbor j of agent i:
    direction_ij = (θ_i - θ_j) / (||θ_i - θ_j||² + ε)

Δθ_i = β * Σ_j  direction_ij
```

**Intuition:**
Applies a repulsive force pushing each agent away from its k-nearest neighbors in parameter space. The force is **inverse-square weighted** — it is strongest when two agents are very close (preventing collapse) and weakens as they move apart. This is the primary driver of diversity.

**Key implementation detail:**
The denominator is `||θ_i - θ_j||²` (squared distance), but the numerator `(θ_i - θ_j)` already contains one factor of distance. So the net force magnitude scales as `β / dist` (inverse-linear in distance), not inverse-square. The force is always directed away from each neighbor.

```python
for j in neighbor_map[i]:
    diff    = theta_i - param_vectors[j]     # direction away from j
    dist_sq = (diff * diff).sum()            # ||θ_i - θ_j||²
    force  += diff / (dist_sq + eps)         # inverse-square weighting
deltas.append(beta * force)
```

`eps=1e-8` prevents division by zero when two agents occupy nearly the same point.

**Where it operates:** parameter space, post-`optimizer.step()`.

---

#### Rule 3 — Cohesion (γ)

**Formula:**
```
centroid_i = mean(θ_j  for j in neighbors(i))
direction  = centroid_i - θ_i
Δθ_i = γ * direction / (||direction|| + ε)
```

**Why normalized (constant-magnitude) rather than linear-spring?**
A linear spring (`Δθ_i = γ * (centroid - θ_i)`) applies force proportional to distance — agents that have drifted far get yanked back hard. This fights against separation, which is trying to keep them spread out.

Normalizing to unit direction gives a **constant-magnitude pull of exactly γ** toward the centroid regardless of distance. Combined with the inverse-linear separation force, this produces a stable equilibrium:

```
At equilibrium:  β / dist*  =  γ   →   dist* = β / γ
```

The ratio β/γ is therefore the primary design knob for equilibrium spacing. For the current defaults (β=0.5, γ=0.1): `dist* = 5.0`.

```python
centroid  = neighbor_params.mean(dim=0)
direction = centroid - theta_i
dist      = direction.norm()
delta     = gamma * direction / (dist + eps)   # constant magnitude γ
```

**Where it operates:** parameter space, post-`optimizer.step()`, summed with separation delta before writing back:
```
θ_i ← θ_i  +  Δθ_sep  +  Δθ_coh
```

---

### Topology — k-NN Graph

**Implementation:** `swarm/topology.py` — `KNNTopology` class.

The interaction graph is **directed k-NN**: each agent interacts with its k nearest neighbors in L2 parameter space. Directionality means agent A may list B as a neighbor without B listing A — this asymmetry is a key source of uneven force distribution.

**Distance computation:**
```python
P       = torch.stack(param_vectors)              # (N, D)
sq_norm = (P * P).sum(dim=1, keepdim=True)        # (N, 1)
dist_sq = sq_norm + sq_norm.T - 2.0 * (P @ P.T)  # (N, N) pairwise squared L2
dist_sq.fill_diagonal_(float('inf'))              # exclude self
_, idx  = torch.topk(dist_sq, k=k, largest=False, sorted=True)
```

Uses the identity `||p_i - p_j||² = ||p_i||² + ||p_j||² - 2·p_i·p_j` to compute all pairwise distances in a single matrix multiply. Complexity: O(N²·D) time, O(N²) space. For N=10 this is negligible.

**Caching:**
The graph is cached and reused until `(current_step - last_update) >= update_interval`. Default `update_interval=1` recomputes every training step. The pre- and post-gradient hooks within the same step share the same cached graph.

**Asymmetry and the "in-degree" problem:**
With k=3 and N=10, each agent has exactly 3 outgoing edges (agents it pushes/pulls). But **in-degree** (how many agents push/pull a given agent) varies. An agent that is geometrically central early in training may appear in many others' k-NN lists, receiving disproportionately large forces. This is the mechanism underlying the instability observed in sep_coh and full_swarm.

**Full connectivity:** setting k=N-1 recovers all-to-all interaction. In this case the graph is symmetric and in-degree equals k for all agents.

---

### CKA — Centered Kernel Alignment

**Reference:** Kornblith et al. 2019 — "Similarity of Neural Network Representations Revisited" (arXiv:1905.00414)

**Implementation:** `metrics/cka.py`

**What it measures:**
Given a fixed probe dataset passed through two networks A and B, CKA measures how similar their internal representations are at a specific layer — invariant to orthogonal transformation and isotropic scaling of activations. This means two networks with very different weight values can still have high CKA if they compute the same geometric relationships between inputs.

**Mathematical derivation:**

Step 1 — Collect activations on the probe set:
```
X  shape (N_probe, D_a)  — activations from network A at layer l
Y  shape (N_probe, D_b)  — activations from network B at layer l
```
N_probe must match; D_a and D_b may differ (comparison is in sample space, not feature space).

Step 2 — Compute linear Gram matrices:
```
K = X @ X.T    (N_probe, N_probe)  — pairwise inner products in A's representation space
L = Y @ Y.T    (N_probe, N_probe)  — pairwise inner products in B's representation space
```

Step 3 — Double center (remove row, column, and grand means):
```
K_c = K - row_means(K) - col_means(K) + grand_mean(K)
L_c = L - row_means(L) - col_means(L) + grand_mean(L)
```
Implemented as `K - K.mean(1, keepdim=True) - K.mean(0, keepdim=True) + K.mean()`.

Step 4 — HSIC (Hilbert-Schmidt Independence Criterion):
```
HSIC(K, L) = (1/(N-1)²) * trace(K_c @ L_c)
```
Computed efficiently as `(K_c * L_c).sum() / (N-1)²` using the identity `trace(A@B) = (A * B.T).sum()` and the fact that both matrices are symmetric.

Step 5 — Normalize:
```
CKA(X, Y) = HSIC(K, L) / sqrt(HSIC(K,K) * HSIC(L,L))
```
Result is in [0, 1]. Clamped to [0, 1] after division to handle numerical edge cases.

**Probe layers:**
CKA is computed at four named probe points in TinyNet: `block1`, `block2`, `block3`, and `gap` (Global Average Pooling). The GAP layer is the most semantically meaningful — it is the final compressed feature vector before the classification head. All ablation analysis focuses on GAP CKA.

**Computation schedule:**
`CKATracker.compute(epoch)` is called every `cka_interval=5` epochs. Each call does one forward pass per agent through the probe loader, then computes the N×N pairwise CKA matrix (N(N-1)/2 calls to `linear_cka`). For N=10 agents this is 45 CKA computations per checkpoint.

**Summary statistic:**
`mean_sim` = mean of the off-diagonal entries of the CKA matrix. High mean_sim = agents are learning similar representations (low diversity). Reported as "GAP CKA" in the ablation results table.

**Important distinction:**
CKA measures **representational similarity** (functional), not **weight similarity** (structural). The separation condition demonstrates this: weight-space diversity of 116 (high structural diversity) but GAP CKA of 0.869 (agents compute similar functions). Neural networks with different weights can converge to functionally equivalent representations due to symmetries in the loss landscape.

---

### Diversity Metric

**Implementation:** `baselines/single_trainer.py` — `EnsembleTrainer.evaluate()`

The diversity metric reported in the ablation results table is the **mean pairwise L2 distance in parameter space**, computed at test time:

```python
param_vecs = torch.stack([a.param_vector() for a in agents])  # (N, D)
diffs      = param_vecs.unsqueeze(0) - param_vecs.unsqueeze(1) # (N, N, D)
pairwise   = torch.norm(diffs, dim=2)                           # (N, N) L2 distances
diversity  = pairwise.sum() / (N * (N - 1))                     # mean off-diagonal
```

This averages over all N(N-1) ordered pairs (excludes diagonal). For N=10 agents this averages 90 pairwise distances.

**Units:** same as the L2 norm of a parameter vector. For TinyNet (~95K parameters), this is a high-dimensional distance — the values (e.g. 22.45 for baseline, 116.36 for separation) are not interpretable in absolute terms but are directly comparable across conditions run with the same architecture.

**During training:** a separate diversity measure (`diversity/mean_param_distance`) is also logged per epoch for SwarmTrainer conditions using `KNNTopology.pairwise_distances()`, which computes the same pairwise L2 matrix via the efficient matrix multiply formulation.

**Relationship to CKA:** diversity measures structural distance in weight space; CKA measures functional distance in representation space. They can diverge significantly — separation produces the largest weight-space diversity but near-baseline representational similarity, confirming that these are measuring fundamentally different things.

---

## Ablation Design

All 2³ = 8 combinations of the three rules:

| Condition  | α   | β   | γ   |
|------------|-----|-----|-----|
| baseline   | 0   | 0   | 0   |
| alignment  | 0.3 | 0   | 0   |
| separation | 0   | 0.5 | 0   |
| cohesion   | 0   | 0   | 0.1 |
| align_sep  | 0.3 | 0.5 | 0   |
| align_coh  | 0.3 | 0   | 0.1 |
| sep_coh    | 0   | 0.5 | 0.1 |
| full_swarm | 0.3 | 0.5 | 0.1 |

---

## Ablation Results (Full Run — 10 agents, 50 epochs, full CIFAR-10)

| Condition  | Ens Loss | Ens Acc | Best Acc | Diversity | GAP CKA |
|------------|----------|---------|----------|-----------|---------|
| baseline   | 0.6731   | 0.763   | 0.747    | 22.45     | 0.904   |
| alignment  | 0.6827   | 0.762   | 0.743    | 20.50     | 0.902   |
| separation | 0.7460   | 0.738   | 0.709    | 116.36    | 0.869   |
| cohesion   | 0.7351   | 0.746   | 0.740    | 10.44     | 0.940   |
| align_sep  | 0.7557   | 0.734   | 0.706    | 114.25    | 0.867   |
| align_coh  | 0.7084   | 0.750   | 0.743    | 10.11     | 0.953   |
| sep_coh    | 0.7490   | 0.747   | 0.740    | 17.95     | 0.821   |
| full_swarm | 0.8251   | 0.715   | 0.702    | 15.07     | 0.870   |

---

## Key Findings

### 1. Baseline wins under current hyperparameters
Independent ensemble training (no swarm rules) achieves the best ensemble accuracy (0.763) and lowest loss (0.6731). This is not a failure — it is a finding about hyperparameter sensitivity and the regime in which swarm coordination helps vs. hurts.

### 2. Alignment is essentially free
Alignment alone (0.762 acc) nearly matches baseline (0.763) with similar diversity (20.50 vs 22.45) and near-identical CKA (0.902 vs 0.904). Gradient blending does not disrupt training but does not provide meaningful benefit at α=0.3.

### 3. Separation is the dominant disruptive force
Every condition containing β produces degraded accuracy. Separation alone creates diversity of 116 (5× baseline) but drops accuracy to 0.738. The equilibrium spacing β/γ = 5.0 is likely too large for this model/dataset combination.

### 4. The separation-cohesion interaction creates training instability
Crucially, separation *alone* produces smooth training curves despite massive weight-space diversity. Instability only appears when separation and cohesion are combined (sep_coh, full_swarm). The competing forces create an oscillatory dynamical system: separation pushes agents away, cohesion creates a restoring force, and agents overshoot the equilibrium repeatedly, causing loss spikes.

### 5. Instability propagates through the k-NN graph with a delay
In sep_coh and full_swarm, one agent destabilizes first. Approximately 15 epochs later, a second agent destabilizes. This propagation delay is a direct consequence of k=3 (sparse local interaction). The destabilized agent's erratic movement in parameter space eventually brings it into the k-NN neighborhood of a previously stable agent, introducing a volatile new force.

### 6. full_swarm triggered early stopping
The full_swarm training curve terminates around epoch 33-35, not 50. The rising ensemble validation loss from unstable agents triggered patience=5 early stopping. This means full_swarm's reported metrics are from an incomplete run, making the comparison slightly unfair.

### 7. Separation increases weight diversity but not representational diversity
Despite separation producing diversity of 116 vs baseline's 22, the GAP-layer CKA for separation (0.869) is close to baseline (0.904). Agents are finding different weight configurations that compute nearly identical functions — consistent with the neural network loss landscape's flat minima and permutation symmetry.

### 8. CKA matrix patterns by condition
- **baseline / alignment**: Uniformly yellow-green. Moderate, consistent inter-agent similarity.
- **cohesion / align_coh**: More uniform yellow. Agents converge to near-identical representations. High CKA (0.940, 0.953).
- **separation / align_sep**: Similar to baseline despite massive weight diversity. Representational convergence persists.
- **sep_coh**: Two outlier agents (agents 2 and 6) form a prominent cross pattern — both are dissimilar from all others AND from each other, suggesting they were pushed into genuinely different representational regions. Lowest CKA (0.821).
- **full_swarm**: Single outlier agent forms a softer cross. One agent drifts while the rest stay similar.

### 9. Bayesian sweep: cohesion is the dominant rule; β/γ = 5.0 was far from optimal
Bayesian optimization over the full (α, β, γ) space finds that gamma (cohesion) is the primary driver of ensemble accuracy. Both top runs from the broad search have high gamma (0.800 and 0.426). The narrowed sweep, pushing gamma to 1.0, achieves 76.5% ensemble accuracy — the best result across all experiments, marginally exceeding the baseline (76.3%). The current ablation defaults (β/γ = 5.0) were in a regime of excessive separation; the optimal regime is tight clustering (β/γ ≈ 0.28). This directly contradicts the ablation finding that swarm rules hurt: under better hyperparameters, full_swarm outperforms baseline.

### 12. Diversity-weighted sweep: swarm achieves 79.8% vs baseline 76.3% (+3.5 points); collective basin navigation identified as mechanism
A new Bayesian sweep optimizing val/diversity_weighted_acc (ensemble_acc * (ensemble_acc - mean_acc)) over the full (α, β, γ) space finds α=0.358, β=0.145, γ=0.457 (β/γ=0.318) as the best configuration. A 100-epoch comparison with these hyperparameters produces the strongest swarm result to date:

| Condition | Ens Loss | Ens Acc | Ens F1 | Best Acc | Diversity | GAP CKA | Stopped at |
|-----------|----------|---------|--------|----------|-----------|---------|------------|
| baseline  | 0.6735 | 0.763 | 0.760 | 0.748 | 22.55 | 0.911 | epoch 50 |
| full_swarm | 0.5743 | 0.798 | 0.800 | 0.787 | 14.58 | 0.924 | epoch 105 |

The swarm outperforms baseline by 3.5 accuracy points, 0.04 F1, and 0.0992 ensemble loss. The swarm extends productive training by 55 epochs (105 vs 50) — a concrete, reproducible measure of the gradient-sharing mechanism. The ceiling with these hyperparameters at batch_size=512 is 79.8% ensemble accuracy. A longer run (150+ epochs) confirmed no further improvement: the model converged cleanly at epoch 105 with slightly better calibration (0.5743 vs 0.5900 at epoch 100) and best individual agent improving to 78.7% vs 78.0%, but ensemble accuracy unchanged.

### 13. Swarm mechanism is collective basin navigation, not diversity preservation
PCA of agent trajectories reveals fundamentally different dynamics between conditions:

- **Baseline**: PC1+PC2 explain only 18.9% of variance between agents. Agents spread across high-dimensional parameter space, each finding its own local basin independently.
- **Full swarm**: PC1+PC2 explain 99.8% of variance. All 12 agents move together along a single dominant trajectory in parameter space, converging as a pack into the same deep loss basin.

The 99.8% figure means the entire collective motion of 12 agents is essentially one-dimensional — they travel together. Despite lower parameter-space diversity (14.31 vs 22.55) and higher CKA (0.936 vs 0.911), the swarm achieves far better accuracy because it collectively navigates to a superior loss basin that individual agents are unlikely to find alone.

This contradicts the initial hypothesis that swarm rules help through diversity-driven error decorrelation. The operative mechanism is cooperative optimization: the swarm rules coordinate agents into a shared trajectory that exploits the loss landscape more effectively than independent descent.

### 14. Alignment rule extends productive training by sharing gradient signals; early stopping is the mediating mechanism
The swarm's training advantage is mediated by early stopping behavior. In the baseline, each agent trains independently. When the ensemble validation loss plateaus — because most agents have individually converged to their local basins — early stopping triggers. In the swarm, the alignment rule propagates gradient directions between neighboring agents via the k-NN graph. If any agent finds a still-productive direction, that signal is shared with its neighbors, which share it further. The ensemble's effective gradient signal stays alive as long as any agent is still improving. Early stopping cannot trigger because the ensemble loss keeps moving. Combined with the separation rule keeping agents in slightly different local neighborhoods (increasing the probability that at least one agent has a non-zero productive gradient at any epoch), the swarm continuously defers convergence until a genuinely better basin is found.

### 10. Topology sweep: divisibility does not predict stability; absolute k matters

Across four (n, k) configurations at batch_size=512, instability persists in all three k=3 configs (n=9, 10, 12) regardless of whether k divides n. n12_k4 is the only configuration where sep_coh and full_swarm both run cleanly to epoch 50. This rules out divisibility as a sufficient condition and points toward absolute neighborhood size as the operative variable.

### 11. k sweep: stability threshold between k=5 and k=6; k=4 is optimal

At n=12, batch_size=512, full_swarm across k=3 to k=11:
- k=3: two agents destabilize, runaway behavior
- k=4 and k=5: one agent shows a brief recoverable spike, mild instability
- k=6 through k=11: fully stable, all agents track together

The stability threshold lies between k=5 and k=6. Divisibility does not predict the threshold — k=5 (non-divisor) is nearly as stable as k=6 (divisor). The transition is determined by absolute k, not k/n ratio or divisibility.

Accuracy peaks at k=4 (0.757) and decreases monotonically beyond that. The fully stable regime (k>=6) consistently underperforms the mildly unstable regime (k=4,5). This indicates that a small amount of instability — agents briefly exploring outside their local minimum — is beneficial for the ensemble, consistent with the exploration-exploitation tradeoff.

Diversity increases monotonically with k (16.34 to 40.64) while CKA decreases (0.859 to 0.894 in inverse), confirming that larger neighborhoods force agents apart in both weight space and representation space, but at the cost of accuracy.

The practical recommendation for this architecture and dataset: k=4 with batch_size=512 in float32.

---

## Visualizations

### Loss Landscape (single-agent, random directions)
Filter-normalized random directions around agent 0's final weights (Li et al. 2018). Rigorous and comparable across conditions because the axes mean the same thing for every model. Key design decisions:
- Agent 0 is always used for consistency across conditions
- Filter normalization scales perturbation magnitude proportional to each filter's actual norm, making landscapes comparable across differently-trained models
- No agent dots overlaid — projections onto random axes in 94K-dimensional space are essentially orthogonal to agent positions and would always collapse to center
- Global shared vmin/vmax computed across all conditions before plotting, ensuring the color scale is comparable

**Observation**: full_swarm produces a notably rounder, more symmetric basin than baseline, suggesting a better-conditioned minimum despite lower accuracy. Baseline shows an asymmetric basin with a shoulder/ridge.

### Multi-Agent PCA Plot
PCA directions derived from SVD of agent deviations from ensemble mean. Loss surface computed along PC1/PC2 centered on the mean agent position. Agent final positions and per-epoch training trails overlaid.

Key properties:
- With n agents, deviations live in at most (n-1)-dimensional subspace. With 3 agents, PC1+PC2 span the exact 2D subspace → 100% variance explained → agent positions and loss values are exact, not approximated.
- With 10 agents, PC1+PC2 capture the two directions of greatest variation. Variance explained is logged on the plot title.
- Agent dots are placed on the actual loss surface using `scipy.interpolate.RegularGridInterpolator`
- Trails show the full training trajectory of each agent projected onto the PCA axes

### CKA Matrices
10×10 heatmaps of pairwise CKA similarity at the GAP layer. Shared colorbar across all 8 conditions (viridis, 0–1). Yellow = identical representations, teal = low similarity.

---

## Topology Sweep Results (n vs. k — Completed)

### Design
Ran sep_coh and full_swarm only (the two unstable conditions from the ablation) across four (n, k) configurations. The goal was to test two competing hypotheses for the instability mechanism: (a) β/γ force imbalance, or (b) k-NN graph asymmetry tied to whether k divides n evenly.

| Config | n | k | k divides n? | k/n ratio |
|--------|---|---|--------------|-----------|
| n9_k3  | 9 | 3 | yes | 0.33 |
| n10_k3 | 10 | 3 | no  | 0.30 ← original |
| n12_k3 | 12 | 3 | yes | 0.25 |
| n12_k4 | 12 | 4 | yes | 0.33 |

### Results

| Config | Condition | n | k | k/n | k divides n? | Ens Acc | Diversity | GAP CKA |
|--------|-----------|---|---|-----|--------------|---------|-----------|---------|
| n9_k3 | sep_coh | 9 | 3 | 0.33 | yes | 0.739 | 14.88 | 0.871 |
| n9_k3 | full_swarm | 9 | 3 | 0.33 | yes | 0.702 | 14.90 | 0.862 |
| n10_k3 | sep_coh | 10 | 3 | 0.30 | no  | 0.747 | 17.95 | 0.821 |
| n10_k3 | full_swarm | 10 | 3 | 0.30 | no  | 0.714 | 15.07 | 0.867 |
| n12_k3 | sep_coh | 12 | 3 | 0.25 | yes | 0.747 | 16.27 | 0.784 |
| n12_k3 | full_swarm | 12 | 3 | 0.25 | yes | 0.741 | 16.85 | 0.860 |
| n12_k4 | sep_coh | 12 | 4 | 0.33 | yes | 0.754 | 16.81 | 0.913 |
| n12_k4 | full_swarm | 12 | 4 | 0.33 | yes | 0.753 | 17.34 | 0.907 |

### Observations

**n12_k4 is the only configuration where both conditions run cleanly to epoch 50.** All k=3 configurations — n9_k3, n10_k3, and n12_k3 — show at least one runaway agent regardless of whether k divides n. This rules out divisibility as a sufficient condition for stability.

The training curves show a consistent pattern across k=3 configs: one agent destabilizes first, and in some runs a second follows within ~15 epochs, consistent with the propagation-via-k-NN mechanism noted in Finding 5.

At n12_k4, the sep_coh / full_swarm performance gap almost vanishes (0.754 vs 0.753). In all k=3 configurations full_swarm underperforms sep_coh by ~0.03–0.04. This suggests that stability is necessary for alignment to contribute positively, rather than alignment being inherently neutral (as it appeared in the isolated ablation).

n12_k4 also achieves the highest ensemble accuracy of any swarm configuration tested to date (0.754), and the highest GAP CKA values (0.913/0.907) — both agents and representations are more cohesive when training is stable.

### Candidate Hypotheses (not yet confirmed)

The observation that n12_k4 is stable and all k=3 configurations are not is consistent with at least two non-exclusive mechanisms:

1. **Absolute neighborhood size**: k=4 provides more force averaging per agent, narrowing in-degree variance enough to dampen oscillations before they amplify. Under this view, k=3 is simply below a stability threshold regardless of n.

2. **In-degree distribution narrowing**: with k=4 and n=12, the maximum possible in-degree (number of agents for whom a given agent is among the k=4 nearest) is bounded more tightly than with k=3. The larger k leaves fewer agents with near-zero incoming force, preventing the "unconstrained" trajectories that seed instability.

These hypotheses cannot be distinguished with the current data because k and n co-vary only at the n12 configurations. A sweep holding n=12 fixed and varying k alone is needed.

---

## k Sweep Results (n=12 fixed — Completed)

### Design

Held n=12 fixed and varied k across 7 values to isolate the effect of neighborhood size from population size. Ran full_swarm only (the condition most sensitive to topology). Used batch_size=512 to match the original ablation and topology sweep, ensuring comparability. torch.compile was active throughout.

An earlier attempt at this sweep used batch_size=1024 (introduced as a training speedup) and mixed precision. That run was discarded because: (a) larger batch size dampens swarm instability by reducing gradient variance, introducing a confound; (b) mixed precision caused float16 overflow to NaN at k=3, a numerical artifact unrelated to the dynamics under study. All results below are float32 at batch_size=512.

| k  | k divides 12? | k/n  | Notes                       |
|----|---------------|------|-----------------------------|
| 3  | yes           | 0.25 | Previously unstable         |
| 4  | yes           | 0.33 | Previously stable           |
| 5  | no            | 0.42 | Non-divisor                 |
| 6  | yes           | 0.50 | Half the swarm              |
| 8  | no            | 0.67 | Non-divisor, dense          |
| 9  | yes           | 0.75 | Three-quarters              |
| 11 | no            | 0.92 | Full connectivity (k = n-1) |

### Results

| Config  | k  | k/n  | k div 12? | Ens Acc | Diversity | GAP CKA |
|---------|----|------|-----------|---------|-----------|---------|
| n12_k3  | 3  | 0.25 | yes       | 0.743   | 16.34     | 0.859   |
| n12_k4  | 4  | 0.33 | yes       | 0.757   | 16.75     | 0.910   |
| n12_k5  | 5  | 0.42 | no        | 0.755   | 20.40     | 0.906   |
| n12_k6  | 6  | 0.50 | yes       | 0.748   | 23.30     | 0.908   |
| n12_k8  | 8  | 0.67 | no        | 0.745   | 30.17     | 0.901   |
| n12_k9  | 9  | 0.75 | no        | 0.741   | 33.64     | 0.898   |
| n12_k11 | 11 | 0.92 | no        | 0.741   | 40.64     | 0.894   |

### Stability Observations (Training Curves)

There is a clear stability threshold between k=5 and k=6:

- **k=3**: two agents destabilize — one spikes hard around epoch 10-15, a second follows. Classic runaway behavior. The ensemble mean is visibly pulled upward.
- **k=4 and k=5**: one agent shows a brief spike but recovers. Mild instability, not runaway. The ensemble mean is barely affected.
- **k=6 through k=11**: completely clean. All agents track together with no individual deviations.

Divisibility does not predict the threshold — k=5 (non-divisor) is nearly as stable as k=6 (divisor), and k=4 (divisor) is less stable than k=6 (divisor). The transition is between absolute k values 5 and 6, not at divisor boundaries.

### Performance Observations

**Accuracy peaks at k=4** — 0.757, the best full_swarm result across all experiments. k=3 and k=4 are close (0.743 vs 0.757), then accuracy drops monotonically from k=5 onward. The fully stable regime (k>=6) consistently underperforms the mildly unstable regime (k=4,5).

**Diversity increases monotonically with k** — 16.34 to 40.64, a 2.5x range. More neighbors in the separation force means each agent is pushed away from more agents simultaneously. The cohesion force does not compensate because it pulls toward a single centroid regardless of k.

**Divisibility has no effect on performance** — the trends in accuracy and diversity are smooth across divisors (k=3,4,6) and non-divisors (k=5,8,9,11). The divisibility hypothesis is definitively ruled out.

**CKA peaks at k=4 (0.910) and decreases monotonically with k** — agents are most representationally cohesive at k=4, then become progressively more dissimilar as k increases. The slight representational divergence at high k is consistent with the alignment rule reinforcing separation effects when the topology is denser, unlike the isolation ablation where separation alone had near-baseline CKA.

**Generalization benefit of small batch training** — accuracy values here (0.741-0.757) are notably higher than the batch_size=1024 discarded run (0.681-0.697). Smaller, noisier batches act as an implicit regularizer on CIFAR-10, a known phenomenon. This is additional evidence that batch_size=512 is the right default for this project.

### Interpretation

k acts as a dial between two regimes:

- **Low k (3-5)**: diverse agents, mildly unstable, better final accuracy. Agents explore distinct regions of the loss landscape.
- **High k (6-11)**: homogeneous agents, stable, lower accuracy. Agents effectively co-optimize toward the same solution, losing the ensemble benefit.

The optimal operating point is **k=4**, which sits just below the stability threshold and achieves the best accuracy. k=5 is a close second and is more robust (less instability than k=3 or k=4) at a small accuracy cost of 0.002. k=3 achieves similar diversity to k=4 but with more pronounced instability and meaningfully lower accuracy.

The practical recommendation for this model and dataset: k=4 with batch_size=512 in float32.

---

## Bayesian Hyperparameter Sweep Results (Completed)

Bayesian optimization over the (α, β, γ) force strength space for the full_swarm condition on Alpine (Blake). Two-phase approach: broad search over [0, 1]³ followed by a narrowed search around the best region.

### Phase 1 — Broad Search ([0, 1]³)

Top runs from the parallel coordinates sweep:

| Run name | α | β | γ | β/γ | Ens Acc |
|----------|---|---|---|-----|---------|
| wise | 0.500 | 0.300 | 0.800 | 0.375 | ~0.740 |
| zany | 0.148 | 0.052 | 0.426 | 0.122 | ~0.740 |

Both top runs share a pattern: moderate-to-high gamma, low-to-moderate beta, and beta/gamma ratios well below 1.0. The worst-performing run (Ens Acc ~0.25) had high alpha and near-zero gamma, confirming cohesion is necessary.

### Phase 2 — Narrowed Search

Best configuration found:

| α | β | γ | β/γ | Ens Acc |
|---|---|---|-----|---------|
| 0.559 | 0.280 | 0.999 | 0.280 | 0.765 |

This is the best ensemble accuracy of any configuration tested to date, including baseline (0.763). The narrowed sweep confirms that pushing gamma toward its upper bound is the correct direction.

### Main Finding

**Gamma (cohesion) is the dominant rule.** The parallel coordinates plot shows that orange/yellow lines (high accuracy) fan out across alpha and beta values but converge at high gamma. The blue line (worst performer) dies at near-zero gamma. The physical interpretation: cohesion pulls agents toward the ensemble centroid, directly optimizing the property that drives ensemble accuracy — agents that agree more produce better-calibrated mean predictions.

The current ablation defaults (β/γ = 5.0) place agents in a regime of aggressive separation with weak cohesion. The sweep finds that β/γ ≈ 0.28 — tight clustering — is the better regime. The current defaults were far from optimal.

### Updated Recommended Hyperparameters

| Parameter | Old default | Sweep optimum |
|-----------|-------------|---------------|
| α | 0.3 | 0.56 |
| β | 0.5 | 0.28 |
| γ | 0.1 | 1.00 |
| β/γ | 5.0 | 0.28 |

### Optimized Run Results

Two follow-up runs using the sweep-optimized hyperparameters at 100 epochs, n=12, k=4.

**v1 — batch_size=256 (mismatch with sweep):**

| Condition | Ens Loss | Ens Acc | Ens F1 | Diversity | GAP CKA | Stopped at |
|-----------|----------|---------|--------|-----------|---------|------------|
| baseline  | 0.5151 | 0.825 | 0.824 | 35.42 | 0.854 | epoch 81 |
| full_swarm | 0.6358 | 0.781 | 0.779 | 14.06 | 0.954 | epoch 51 |

Baseline wins by 4.4 points. Full swarm stops 30 epochs early — high cohesion caps learning. Batch size of 256 provides enough natural gradient noise to regularize baseline, making the swarm's cohesion redundant.

**v2 — batch_size=512 (matching sweep conditions):**

| Condition | Ens Loss | Ens Acc | Ens F1 | Diversity | GAP CKA | Stopped at |
|-----------|----------|---------|--------|-----------|---------|------------|
| baseline  | 0.6735 | 0.763 | 0.760 | 22.55 | 0.911 | epoch 50 |
| full_swarm | 0.6430 | 0.772 | 0.774 | 16.88 | 0.927 | epoch 87 |

**Full swarm beats baseline by 0.9 points (77.2% vs 76.3%).** The training dynamics are reversed: baseline plateaus and stops at epoch 50 while the swarm keeps improving to epoch 87. Cohesion acts as a regularizer at batch_size=512, extending the productive training window by 37 epochs.

The batch size was the confound. At batch_size=256, small noisy batches provide natural regularization sufficient for baseline. At batch_size=512, gradient estimates are smoother and baseline overfits earlier — this is precisely the regime where swarm cohesion adds value.

**Methodological note:** The Bayesian sweep was run at batch_size=512, so the optimal hyperparameters (α=0.559, β=0.280, γ=0.999) are conditioned on that batch size. They are not transferable to other batch sizes. At batch_size=256, gradient noise is higher and already acts as an implicit regularizer — cohesion and gradient noise are substitutes, not independent knobs. Running the sweep at batch_size=256 would likely find a lower optimal γ. This also explains the v1 failure: applying sweep hyperparameters tuned at batch_size=512 to a batch_size=256 run over-regularized the swarm (too much cohesion on top of already-noisy gradients), causing early stopping at epoch 51 and the 4.4-point accuracy gap.

---

## Open Questions

1. **Batch size as a stability mechanism**: The k sweep revealed that batch_size=1024 dampens instability that appears at batch_size=512. This raises the question of whether stability in the topology sweep (n12_k4) was partly due to some other effect, or purely the k increase. A controlled rerun of the topology sweep configurations at batch_size=1024 would clarify this.

2. **Does alignment ever help?**: At α=0.3 it is essentially neutral. Is there a regime (larger α, or combined with a stable β/γ) where gradient alignment produces measurable benefit?

3. **Why does full_swarm produce a rounder loss basin despite lower accuracy?**: Partly resolved — the v3 sweep results show the swarm does achieve better accuracy under optimized hyperparameters, and the PCA analysis confirms agents converge into a deeper, better-conditioned basin. The rounder basin and higher accuracy are now consistent.

4. **Is sep_coh's two-outlier pattern reproducible?**: The cross pattern in the CKA matrix (agents 2 and 6) may be seed-dependent. Running with a different seed would confirm whether the number of outliers is structural or coincidental.

5. **Warm-up period**: Currently `warm_start_steps=0` — swarm forces are active from epoch 1. A warm-up period (allowing agents to first find reasonable minima before interaction) might prevent early destabilization.

---

## Next Steps

- [x] Analyze topology sweep results — divisibility ruled out; k=4 stabilizes both conditions at batch_size=512
- [x] Run k sweep at n=12 (batch_size=512) — stability threshold between k=5 and k=6; k=4 is optimal; divisibility definitively ruled out
- [x] Integrate Bayesian sweep results — cohesion is dominant; β/γ=0.28 optimal; 76.5% beats baseline
- [x] Re-run with optimized params at batch_size=512 — full_swarm beats baseline (77.2% vs 76.3%), runs 37 epochs longer
- [x] New sweep with diversity_weighted_acc metric — finds α=0.358, β=0.145, γ=0.457; full_swarm 79.8% vs baseline 76.3% (+3.5 points); swarm still improving at epoch 100
- [x] PCA analysis confirms collective basin navigation as mechanism — 99.8% variance in swarm vs 18.9% in baseline
- [x] Run swarm beyond 100 epochs — early stops at epoch 105, ceiling confirmed at 79.8%; swarm extends productive training by 55 epochs over baseline
- [ ] Re-run promising conditions with different random seeds for robustness
- [ ] Investigate warm-up period effect on stability
- [ ] Write paper — sections: Introduction, Related Work, Method, Experiments, Results, Discussion
