# Swarm-Optimized Neural Network Ensembles

**Multi-Agent Systems — Final Project**

Applies Boids-style swarm coordination rules to neural network ensemble training. Each network in the ensemble is treated as a swarm agent that interacts with its k-nearest neighbors in parameter space, with the goal of improving ensemble diversity and generalization over independent training.

---

## Project Structure

```
.
├── models/                     # Model architecture
├── baselines/                  # Independent ensemble trainer (no swarm)
├── swarm/                      # Swarm trainer, interaction rules, k-NN topology
├── metrics/                    # CKA and diversity metrics
├── experiments/                # Ablation runner, single-run engine, hyperparam sweep
├── visualization/              # Loss landscape and diagnostic plots
├── notebooks/
│   └── swarm_experiment.ipynb  # Colab orchestration notebook
└── requirements.txt
```

---

## Setup

```bash
git clone https://github.com/blakedehaas/multi-agent-ml.git
cd multi-agent-ml
pip install -r requirements.txt
```

---

## Running

Open `notebooks/swarm_experiment.ipynb` in Google Colab. The notebook handles environment setup, GPU configuration, and experiment orchestration. All logic lives in the `.py` modules — the notebook is a thin driver.

---

## Dependencies

PyTorch · torchvision · NumPy · SciPy · scikit-learn · Matplotlib · tqdm · Weights & Biases
