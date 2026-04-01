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

## Ongoing Experiments

### Topology Sweep (in progress)
Testing whether the k-NN neighborhood size being a factor of n_agents affects stability. Runs only sep_coh and full_swarm (the unstable conditions) across four (n, k) configurations:

| Config | n | k | n mod k | k/n ratio |
|--------|---|---|---------|-----------|
| n9_k3  | 9 | 3 | 0 ✅ | 0.33 |
| n10_k3 | 10 | 3 | 1 ❌ | 0.30 ← current |
| n12_k3 | 12 | 3 | 0 ✅ | 0.25 |
| n12_k4 | 12 | 4 | 0 ✅ | 0.33 |

Hypothesis: instability may be tied to asymmetric k-NN graphs when k does not divide n evenly, rather than purely to the β/γ force imbalance.

### Bayesian Hyperparameter Sweep
Bayesian optimization over the (α, β, γ) force strength space for the full_swarm condition. Will explore whether a different β/γ ratio can recover the benefit of swarm coordination without inducing instability. Focus on much smaller β (0.05–0.2 range).

---

## Open Questions

1. **Force balance vs. topology**: Is the instability primarily caused by the β/γ ratio (force imbalance) or by the k/n_agents ratio (graph asymmetry)? The topology sweep will help answer this.

2. **Does alignment ever help?**: At α=0.3 it is essentially neutral. Is there a regime (larger α, or combined with a stable β/γ) where gradient alignment produces measurable benefit?

3. **Why does full_swarm produce a rounder loss basin despite lower accuracy?**: A rounder basin typically implies better generalization. The contradiction between basin geometry and test accuracy deserves investigation.

4. **Is sep_coh's two-outlier pattern reproducible?**: The cross pattern in the CKA matrix (agents 2 and 6) may be seed-dependent. Running with a different seed would confirm whether the number of outliers is structural or coincidental.

5. **Warm-up period**: Currently `warm_start_steps=0` — swarm forces are active from epoch 1. A warm-up period (allowing agents to first find reasonable minima before interaction) might prevent early destabilization.

---

## Next Steps

- [ ] Analyze topology sweep results — determine if k/n divisibility matters
- [ ] Integrate Bayesian sweep results from teammate
- [ ] Re-run promising conditions with different random seeds for robustness
- [ ] Investigate warm-up period effect on stability
- [ ] Write paper — sections: Introduction, Related Work, Method, Experiments, Results, Discussion
