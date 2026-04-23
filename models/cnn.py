"""
models/cnn.py

TinyNet: a lightweight CNN designed for CIFAR-10 under multi-agent constraints.

Design goals:
  - Small enough to run 10 simultaneous copies on one A100 (~200K params/agent)
  - Named intermediate representations for CKA probing
  - Conv filters accessible by name for filter-normalized loss landscape visualization
  - Configurable num_classes so the same architecture works for any dataset

Architecture:
    Input: (B, 3, 32, 32)
    Block 1: Conv(3→32)  → GN(4) → ReLU → MaxPool  →  (B, 32, 16, 16)
    Block 2: Conv(32→64) → GN(4) → ReLU → MaxPool  →  (B, 64,  8,  8)
    Block 3: Conv(64→128)→ GN(4) → ReLU → MaxPool  →  (B, 128,  4,  4)
    GAP:     GlobalAvgPool                           →  (B, 128)
    Head:    Linear(128 → num_classes)               →  (B, num_classes)

    Total parameters: ~95K (well within budget for 10 agents on one A100)

Normalization: GroupNorm(num_groups=4) instead of BatchNorm2d.
    GroupNorm normalizes over channel groups per sample with no running
    stats. It performs better than BatchNorm at the small effective batch
    sizes that arise when training 10 agents over a shared mini-batch,
    and avoids instability from stale running statistics early in training.

CKA probe points (returned by forward_with_probes()):
    'block1'  - post-ReLU feature map, shape (B, 32, 16, 16) -> flattened to (B, 8192)
    'block2'  - post-ReLU feature map, shape (B, 64,  8,  8) -> flattened to (B, 4096)
    'block3'  - post-ReLU feature map, shape (B, 128, 4,  4) -> flattened to (B, 2048)
    'gap'     - post-GAP vector,       shape (B, 128)
"""
from typing import Dict

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Conv -> GroupNorm -> ReLU -> MaxPool block."""

    def __init__(
        self,
        in_channels:  int,
        out_channels: int,
        pool:         bool = True,
        num_groups:   int  = 4,
    ) -> None:
        super().__init__()
        # Named attribute so named_conv_filters() can access it safely by name
        # rather than by fragile positional index into a Sequential.
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) if pool else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.relu(self.norm(self.conv(x))))


class TinyNet(nn.Module):
    """
    Lightweight CNN for CIFAR-10 (or any 32x32 image dataset).

    Parameters
    ----------
    num_classes : int
        Number of output classes. Default 10 for CIFAR-10.
    dropout : float
        Dropout probability applied before the classifier head.
        Default 0.3 -- light regularization, does not dominate at 10 agents.
    """

    def __init__(self, num_classes: int = 10, dropout: float = 0.3) -> None:
        super().__init__()

        self.block1 = ConvBlock(3,   32,  pool=True)   # -> (B, 32, 16, 16)
        self.block2 = ConvBlock(32,  64,  pool=True)   # -> (B, 64,  8,  8)
        self.block3 = ConvBlock(64,  128, pool=True)   # -> (B, 128,  4,  4)

        self.gap     = nn.AdaptiveAvgPool2d(1)          # -> (B, 128, 1, 1)
        self.dropout = nn.Dropout(p=dropout)
        self.head    = nn.Linear(128, num_classes)

    # ------------------------------------------------------------------
    # Standard forward pass
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.gap(x).flatten(1)   # (B, 128)
        x = self.dropout(x)
        return self.head(x)

    # ------------------------------------------------------------------
    # CKA probe forward -- returns logits + intermediate representations
    # ------------------------------------------------------------------
    def forward_with_probes(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Run forward pass and collect intermediate representations for CKA.

        Returns
        -------
        logits : Tensor  shape (B, num_classes)
        probes : dict mapping probe name -> flat representation (B, D)
            'block1' : (B, 32*16*16) = (B, 8192)
            'block2' : (B, 64*8*8)   = (B, 4096)
            'block3' : (B, 128*4*4)  = (B, 2048)
            'gap'    : (B, 128)

        Notes
        -----
        Representations are detached from the compute graph -- they are
        only used as inputs to CKA, never backpropagated through.
        """
        probes: Dict[str, torch.Tensor] = {}

        x = self.block1(x)
        probes['block1'] = x.flatten(1).detach()  # (B, 8192)

        x = self.block2(x)
        probes['block2'] = x.flatten(1).detach()  # (B, 4096)

        x = self.block3(x)
        probes['block3'] = x.flatten(1).detach()  # (B, 2048)

        x = self.gap(x).flatten(1)
        probes['gap'] = x.detach()                # (B, 128)

        x = self.dropout(x)
        logits = self.head(x)

        return logits, probes

    # ------------------------------------------------------------------
    # Filter access -- used by filter-normalized loss landscape
    # ------------------------------------------------------------------
    def named_conv_filters(self) -> dict[str, torch.Tensor]:
        """
        Return a dict of conv weight tensors keyed by layer name.

        Each tensor has shape (out_channels, in_channels, kH, kW).
        Used to compute per-filter norms for landscape normalization:

            d_hat[l,f] = d[l,f] / ||d[l,f]|| * ||W[l,f]||

        Returns
        -------
        dict:
            'block1.conv' : Tensor (32,  3,  3, 3)
            'block2.conv' : Tensor (64,  32, 3, 3)
            'block3.conv' : Tensor (128, 64, 3, 3)
        """
        return {
            'block1.conv': self.block1.conv.weight,
            'block2.conv': self.block2.conv.weight,
            'block3.conv': self.block3.conv.weight,
        }

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------
    def param_count(self) -> int:
        """Total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
