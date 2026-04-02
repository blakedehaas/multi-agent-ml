"""
baselines/single_trainer.py

Single-agent wrapper and independent ensemble baseline trainer.

Class hierarchy intended for extension:

    EnsembleTrainer (abstract)
    └── IndependentEnsembleTrainer  ← baseline, hooks are no-ops
    └── SwarmTrainer (swarm/trainer.py)
            overrides pre_gradient_step()   → gradient alignment
            overrides post_gradient_step()  → separation + cohesion
"""

import copy
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Optional

import torch
import torch.nn as nn
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class TrainingConfig:
    """
    Hyperparameters shared across all trainer types.

    Attributes:
        lr:               Learning rate for the base optimizer.
        optimizer:        "adam" or "sgd".
        momentum:         SGD momentum (ignored for Adam).
        weight_decay:     L2 regularization coefficient.
        warm_start_steps: Number of global steps to run independently
                          before swarm rules activate (α = β = γ = 0
                          during warm-start regardless of their values).
        device:           Torch device string, e.g. "cpu" or "cuda".
        seed:             Optional RNG seed for weight initialization.
    """
    lr: float = 1e-3
    optimizer: str = "adam"
    momentum: float = 0.9
    weight_decay: float = 0.0
    warm_start_steps: int = 0
    device: str = "cpu"
    seed: Optional[int] = None


# ---------------------------------------------------------------------------
# Early stopping
# ---------------------------------------------------------------------------

class EarlyStopping:
    """
    Monitors a scalar metric and signals when training should stop.

    Stops when the metric has not improved by more than `min_delta`
    for `patience` consecutive epochs.

    Usage
    -----
    Instantiate once before training, then call .step(metric) at the
    end of each epoch. Returns True when training should stop.

        stopper = EarlyStopping(patience=3)
        # inside epoch callback:
        if stopper.step(val_loss):
            return True   # signals train() to break

    Parameters
    ----------
    patience  : int   — epochs to wait after last improvement. Default 3.
    min_delta : float — minimum change to count as an improvement. Default 0.
    """

    def __init__(self, patience: int = 3, min_delta: float = 0.0) -> None:
        self.patience  = patience
        self.min_delta = min_delta
        self.best      = float('inf')
        self.counter   = 0

    def step(self, metric: float) -> bool:
        """
        Update state with the latest metric value.

        Returns True if training should stop, False otherwise.
        """
        if metric < self.best - self.min_delta:
            self.best    = metric
            self.counter = 0
        else:
            self.counter += 1

        return self.counter >= self.patience


# ---------------------------------------------------------------------------
# Single-agent wrapper
# ---------------------------------------------------------------------------

class AgentTrainer:
    """
    Wraps a single nn.Module with its optimizer and bookkeeping state.

    AgentTrainer is the atomic unit that EnsembleTrainer coordinates.
    It exposes helpers for reading and writing flat parameter/gradient
    vectors, which the swarm rules operate on.
    """

    def __init__(self, model: nn.Module, config: TrainingConfig):
        self.model = model.to(config.device)
        self.config = config
        self.device = config.device
        self.step_count: int = 0
        self.optimizer = self._build_optimizer()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_optimizer(self) -> torch.optim.Optimizer:
        params = self.model.parameters()
        if self.config.optimizer == "adam":
            return torch.optim.Adam(
                params,
                lr=self.config.lr,
                weight_decay=self.config.weight_decay,
            )
        if self.config.optimizer == "sgd":
            return torch.optim.SGD(
                params,
                lr=self.config.lr,
                momentum=self.config.momentum,
                weight_decay=self.config.weight_decay,
            )
        raise ValueError(f"Unknown optimizer: {self.config.optimizer!r}")

    # ------------------------------------------------------------------
    # Parameter-space helpers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def param_vector(self) -> torch.Tensor:
        """Return a flat clone of all parameters as a 1-D tensor."""
        return parameters_to_vector(self.model.parameters()).clone()

    @torch.no_grad()
    def set_param_vector(self, vec: torch.Tensor) -> None:
        """Write a flat 1-D parameter vector back into the model in-place."""
        vector_to_parameters(vec.to(self.device), self.model.parameters())

    @torch.no_grad()
    def grad_vector(self) -> Optional[torch.Tensor]:
        """
        Return a flat concatenation of all .grad tensors.
        Returns None if backward() has not been called yet.
        """
        grads = [
            p.grad.flatten()
            for p in self.model.parameters()
            if p.grad is not None
        ]
        return torch.cat(grads) if grads else None

    @torch.no_grad()
    def set_grad_vector(self, vec: torch.Tensor) -> None:
        """
        Write a flat 1-D gradient vector back into each parameter's .grad.
        Allocates .grad tensors if they do not yet exist.
        """
        offset = 0
        for p in self.model.parameters():
            numel = p.numel()
            chunk = vec[offset : offset + numel].view_as(p)
            if p.grad is None:
                p.grad = chunk.clone().to(self.device)
            else:
                p.grad.copy_(chunk)
            offset += numel


# ---------------------------------------------------------------------------
# Abstract ensemble base
# ---------------------------------------------------------------------------

class EnsembleTrainer(ABC):
    """
    Abstract base class for all ensemble trainers.

    The training step is factored into four ordered stages so subclasses
    can inject behavior at the right level without rewriting the loop:

        1. forward + backward for every agent  (no optimizer step yet)
        2. pre_gradient_step()                 ← override for alignment
        3. optimizer.step() for every agent
        4. post_gradient_step()                ← override for sep + coh

    The warm_start_steps field in TrainingConfig gates stages 2 and 4:
    hooks are skipped until that many global steps have elapsed.

    IndependentEnsembleTrainer (below) leaves both hooks as no-ops,
    giving the α = β = γ = 0 baseline.  SwarmTrainer (swarm/trainer.py)
    overrides them to implement the full Boids update.
    """

    def __init__(self, agents: list[AgentTrainer], criterion: nn.Module):
        if not agents:
            raise ValueError("EnsembleTrainer requires at least one agent.")
        self.agents = agents
        self.criterion = criterion
        self._global_step: int = 0

        # Mixed precision — one GradScaler per agent.
        # Disabled automatically when not on CUDA (CPU/MPS runs unaffected).
        self._use_amp = agents[0].device != 'cpu'
        self._scalers = [
            torch.cuda.amp.GradScaler(enabled=self._use_amp)
            for _ in agents
        ]

    # ------------------------------------------------------------------
    # Hooks — override in subclasses
    # ------------------------------------------------------------------

    @abstractmethod
    def pre_gradient_step(self) -> None:
        """
        Called after backward() for all agents, before any optimizer.step().

        At this point every agent has fresh .grad tensors.
        Use AgentTrainer.grad_vector() / set_grad_vector() to read and
        write gradients for gradient-level rules (e.g. alignment).
        """

    @abstractmethod
    def post_gradient_step(self) -> None:
        """
        Called after optimizer.step() for all agents.

        At this point parameters have been updated by the optimizer.
        Use AgentTrainer.param_vector() / set_param_vector() to read and
        write parameters for position-level rules (e.g. separation, cohesion).
        """

    # ------------------------------------------------------------------
    # Core training loop
    # ------------------------------------------------------------------

    def train_step(self, batch: tuple[torch.Tensor, torch.Tensor]) -> list[float]:
        """
        One full update for all agents on a single mini-batch.

        Stages
        ------
        1. Forward + backward for every agent (sequential)
        2. pre_gradient_step()  ← alignment hook (swarm only)
        3. optimizer.step() for every agent
        4. post_gradient_step() ← separation + cohesion hook (swarm only)

        Returns
        -------
        List of per-agent scalar loss values (Python floats).
        """
        device = self.agents[0].device
        X = batch[0].to(device)
        y = batch[1].to(device)

        # Stage 1: forward + backward for each agent (mixed precision)
        losses = []
        for i, agent in enumerate(self.agents):
            agent.optimizer.zero_grad()
            with torch.autocast('cuda', enabled=self._use_amp):
                loss = self.criterion(agent.model(X), y)
            self._scalers[i].scale(loss).backward()
            losses.append(loss.item())

        # Stages 2–4 are gated by the warm-start schedule
        swarm_active = self._global_step >= self.agents[0].config.warm_start_steps

        if swarm_active:
            # Unscale gradients back to float32 before swarm rules read them.
            # Without this, grad_vector() would return scaled values and the
            # alignment rule would compute incorrect blended gradients.
            for i, agent in enumerate(self.agents):
                self._scalers[i].unscale_(agent.optimizer)
            self.pre_gradient_step()

        for i, agent in enumerate(self.agents):
            self._scalers[i].step(agent.optimizer)
            self._scalers[i].update()
            agent.optimizer.zero_grad()
            agent.step_count += 1

        if swarm_active:
            self.post_gradient_step()

        self._global_step += 1
        return losses

    def train(
        self,
        dataloader: DataLoader,
        epochs: int,
        callback: Optional[Callable[[int, dict], None]] = None,
    ) -> list[dict]:
        """
        Train for `epochs` full passes over `dataloader`.

        Args:
            dataloader: Yields (X, y) tuples.
            epochs:     Number of epochs to run.
            callback:   Optional fn(epoch, metrics_dict) called at each
                        epoch end — useful for logging or early stopping.

        Returns:
            List of per-epoch metric dicts, each containing:
              - "epoch"       : int
              - "agent_losses": list of mean per-agent loss for that epoch
              - "mean_loss"   : mean across all agents
        """
        history: list[dict] = []

        epoch_bar = tqdm(range(epochs), desc='Training', unit='epoch')

        for epoch in epoch_bar:
            epoch_losses: list[list[float]] = [[] for _ in self.agents]

            batch_bar = tqdm(dataloader, desc=f'  Epoch {epoch+1:3d}', leave=False, unit='batch')
            for batch in batch_bar:
                step_losses = self.train_step(batch)
                for i, loss in enumerate(step_losses):
                    epoch_losses[i].append(loss)
                batch_bar.set_postfix(loss=f'{sum(step_losses)/len(step_losses):.4f}')

            agent_means = [sum(ls) / len(ls) for ls in epoch_losses]
            metrics = {
                "epoch": epoch,
                "agent_losses": agent_means,
                "mean_loss": sum(agent_means) / len(agent_means),
            }
            history.append(metrics)

            if callback is not None:
                stop = callback(epoch, metrics)
                if stop:
                    tqdm.write(f'Early stopping triggered at epoch {epoch}.')
                    break

            epoch_bar.set_postfix(train=f'{metrics["mean_loss"]:.4f}')

        return history

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> dict:
        """
        Compute per-agent and ensemble metrics over a held-out dataloader.

        Returns a dict with:
          - individual_losses : list[float], one per agent
          - mean_loss         : mean individual loss
          - best_loss         : best individual loss (lowest loss)
          - ensemble_loss     : loss of the mean prediction across agents
          - individual_accs   : list[float], accuracy per agent (0–1)
          - mean_acc          : mean accuracy across agents
          - best_acc          : best individual accuracy
          - ensemble_acc      : accuracy of the mean prediction across agents
          - diversity         : mean pairwise L2 distance in parameter space
        """
        all_preds: list[list[torch.Tensor]] = [[] for _ in self.agents]
        all_targets: list[torch.Tensor] = []

        for batch in dataloader:
            X, y = batch
            all_targets.append(y.cpu())
            for i, agent in enumerate(self.agents):
                pred = agent.model(X.to(agent.device)).cpu()
                all_preds[i].append(pred)

        targets = torch.cat(all_targets)
        preds = [torch.cat(p) for p in all_preds]

        individual_losses = [self.criterion(p, targets).item() for p in preds]
        ensemble_pred = torch.stack(preds).mean(dim=0)
        ensemble_loss = self.criterion(ensemble_pred, targets).item()

        # Accuracy: fraction of correct predictions
        individual_accs = [
            (p.argmax(dim=1) == targets).float().mean().item() for p in preds
        ]
        ensemble_acc = (ensemble_pred.argmax(dim=1) == targets).float().mean().item()

        # Mean pairwise L2 distance in parameter space
        param_vecs = torch.stack([a.param_vector() for a in self.agents])  # (N, D)
        diffs = param_vecs.unsqueeze(0) - param_vecs.unsqueeze(1)           # (N, N, D)
        pairwise = torch.norm(diffs, dim=2)                                  # (N, N)
        n = len(self.agents)
        diversity = pairwise.sum().item() / max(n * (n - 1), 1)

        return {
            "individual_losses": individual_losses,
            "mean_loss":         sum(individual_losses) / len(individual_losses),
            "best_loss":         min(individual_losses),
            "ensemble_loss":     ensemble_loss,
            "individual_accs":   individual_accs,
            "mean_acc":          sum(individual_accs) / len(individual_accs),
            "best_acc":          max(individual_accs),
            "ensemble_acc":      ensemble_acc,
            "diversity":         diversity,
        }

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def snapshot(self) -> list[dict]:
        """
        Return a deep copy of each agent's state_dict.
        Useful for checkpointing before an A/B experiment split.
        """
        return [copy.deepcopy(a.model.state_dict()) for a in self.agents]

    def restore(self, snapshots: list[dict]) -> None:
        """Load a list of state_dicts (from snapshot()) back into agents."""
        if len(snapshots) != len(self.agents):
            raise ValueError("Snapshot length does not match number of agents.")
        for agent, state in zip(self.agents, snapshots):
            agent.model.load_state_dict(state)


# ---------------------------------------------------------------------------
# Concrete baseline
# ---------------------------------------------------------------------------

class IndependentEnsembleTrainer(EnsembleTrainer):
    """
    Baseline: N agents trained with standard gradient descent, no interaction.

    Both hooks are explicit no-ops (α = β = γ = 0).
    This is the (0, 0, 0) cell in the ablation matrix.
    """

    def pre_gradient_step(self) -> None:
        pass

    def post_gradient_step(self) -> None:
        pass
