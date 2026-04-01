"""
data/cifar.py

CIFAR-10 data loading for local development and Colab A100 runs.

Entry points
------------
    get_cifar10_loaders()   →  (train_loader, val_loader, test_loader)
    get_probe_loader()      →  fixed 512-sample loader for CKA / landscape viz

Two modes
---------
    Local dev  : pass subset_size=5000  → 5k train, 1k val, fast iteration
    Full run   : pass subset_size=None  → 50k train, 10k test, full CIFAR-10

Augmentation
------------
    Train  : Resize(32) + RandomHorizontalFlip + Normalize
    Val    : Normalize only  (stable, reproducible eval metrics)
    Probe  : Normalize only  (fixed seed, never shuffled — comparable across ckpts)

CIFAR-10 channel statistics (computed over full training set):
    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2470, 0.2435, 0.2616)
"""

import torch
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms
from pathlib import Path


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2470, 0.2435, 0.2616)

# Default download location — sits next to this file: data/cifar10/
_DEFAULT_DATA_ROOT = Path(__file__).parent / "cifar10"


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

def _train_transform() -> transforms.Compose:
    """
    Training transform: resize (no-op for CIFAR-10) + horizontal flip + normalize.

    RandomCrop is intentionally omitted. For this experiment the differences
    between agents should come from swarm rules, not from per-agent spatial
    jitter. A horizontal flip is kept because it is physically meaningful
    (a flipped cat is still a cat) and adds negligible stochasticity.
    """
    return transforms.Compose([
        transforms.Resize(32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])


def _eval_transform() -> transforms.Compose:
    """Clean transform for val / test / probe splits — no augmentation."""
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def get_cifar10_loaders(
    data_root: str | Path = _DEFAULT_DATA_ROOT,
    batch_size: int = 128,
    subset_size: int | None = 5000,
    val_fraction: float = 0.1,
    num_workers: int = 0,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Return (train_loader, val_loader, test_loader) for CIFAR-10.

    The dataset is downloaded automatically on first call to `data_root`.

    Parameters
    ----------
    data_root : str or Path
        Directory where CIFAR-10 will be downloaded / read from.
        Defaults to data/cifar10/ next to this file.
        On Colab, set to '/content/data' or '/content/drive/MyDrive/data'.

    batch_size : int
        Mini-batch size for all loaders. Default 128.

    subset_size : int or None
        If set, use only this many samples from the CIFAR-10 training set.
        The subset is drawn with a fixed seed for reproducibility.
        Pass None to use the full 50k training set (for Colab runs).

        Recommended:
            Local dev  →  subset_size=5000   (~10s/epoch on CPU)
            Colab A100 →  subset_size=None   (full dataset)

    val_fraction : float
        Fraction of the (possibly subsetted) training data to hold out
        for validation. Default 0.1 → 10% val, 90% train.

    num_workers : int
        DataLoader worker processes. Default 2.
        Set to 0 for debugging (avoids multiprocessing issues).
        On Colab A100, set to 4 for best throughput.

    seed : int
        RNG seed for the train/val split and subset selection.

    Returns
    -------
    train_loader : DataLoader   (augmented, shuffled)
    val_loader   : DataLoader   (clean, not shuffled)
    test_loader  : DataLoader   (clean, not shuffled, official CIFAR-10 test set)
    """
    data_root = Path(data_root)

    # ── Download / load raw datasets ──────────────────────────────────────
    # Train split uses augmentation; test and val use clean eval transform.
    # We load the training set twice with different transforms so we can
    # apply augmentation only to the train portion after the val split.
    full_train_aug   = datasets.CIFAR10(data_root, train=True,  download=True,
                                        transform=_train_transform())
    full_train_clean = datasets.CIFAR10(data_root, train=True,  download=True,
                                        transform=_eval_transform())
    test_set         = datasets.CIFAR10(data_root, train=False, download=True,
                                        transform=_eval_transform())

    # ── Optional subsetting ───────────────────────────────────────────────
    rng = torch.Generator().manual_seed(seed)

    if subset_size is not None:
        n_total = len(full_train_aug)
        if subset_size > n_total:
            raise ValueError(
                f"subset_size={subset_size} exceeds available training samples ({n_total})."
            )
        indices = torch.randperm(n_total, generator=rng)[:subset_size].tolist()
        train_pool_aug   = Subset(full_train_aug,   indices)
        train_pool_clean = Subset(full_train_clean, indices)
    else:
        train_pool_aug   = full_train_aug
        train_pool_clean = full_train_clean

    # ── Train / val split ─────────────────────────────────────────────────
    n_pool   = len(train_pool_aug)
    n_val    = max(1, int(n_pool * val_fraction))
    n_train  = n_pool - n_val

    # random_split on the augmented pool gives us train indices;
    # we apply the same indices to the clean pool for val.
    train_subset, val_subset_aug = random_split(
        train_pool_aug, [n_train, n_val], generator=rng
    )

    # Val uses clean (no augmentation) transform — re-index the clean pool
    val_indices = val_subset_aug.indices if hasattr(val_subset_aug, 'indices') else list(range(n_train, n_pool))
    val_subset  = Subset(train_pool_clean, val_indices)

    # ── Build DataLoaders ─────────────────────────────────────────────────
    loader_kwargs = dict(num_workers=num_workers, pin_memory=(num_workers > 0))

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        generator=rng,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        **loader_kwargs,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        **loader_kwargs,
    )

    return train_loader, val_loader, test_loader


# ---------------------------------------------------------------------------
# CKA / landscape probe loader
# ---------------------------------------------------------------------------

def get_probe_loader(
    data_root: str | Path = _DEFAULT_DATA_ROOT,
    n_samples: int = 512,
    batch_size: int = 512,
    seed: int = 0,
) -> DataLoader:
    """
    Return a fixed, non-shuffled loader over a held-out probe set.

    This loader is used for:
      - CKA representational similarity: probe activations must come from
        the same fixed images at every checkpoint so Gram matrices are
        comparable across time and across agents.
      - Filter-normalized loss landscape: the loss is evaluated on a fixed
        set so landscape plots are comparable across runs.

    The probe set is drawn from the CIFAR-10 test set (never seen during
    training) with a fixed seed, and uses the clean eval transform
    (no augmentation) for reproducibility.

    Parameters
    ----------
    data_root : str or Path
        Same root used for get_cifar10_loaders().
    n_samples : int
        Number of probe images. Default 512.
        Must be <= 10000 (CIFAR-10 test set size).
    batch_size : int
        Batch size for the probe loader. Default 512 (loads all at once).
    seed : int
        Fixed seed for probe index selection. Never change this — changing
        it invalidates all previously computed CKA / landscape results.

    Returns
    -------
    DataLoader (not shuffled, fixed order, same images every call)
    """
    data_root = Path(data_root)

    test_set = datasets.CIFAR10(data_root, train=False, download=True,
                                transform=_eval_transform())

    if n_samples > len(test_set):
        raise ValueError(
            f"n_samples={n_samples} exceeds test set size ({len(test_set)})."
        )

    rng     = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(test_set), generator=rng)[:n_samples].tolist()
    probe_set = Subset(test_set, indices)

    return DataLoader(
        probe_set,
        batch_size=batch_size,
        shuffle=False,           # NEVER shuffle — order must be identical every call
        num_workers=0,
    )


# ---------------------------------------------------------------------------
# Dataset info utility
# ---------------------------------------------------------------------------

def dataset_stats(loader: DataLoader) -> dict:
    """
    Return basic statistics about a DataLoader for sanity checking.

    Returns dict with:
        n_samples   : total number of samples
        n_batches   : number of batches
        batch_size  : configured batch size
        class_dist  : dict mapping class_idx -> count (label distribution)
    """
    n_samples  = len(loader.dataset)
    n_batches  = len(loader)
    batch_size = loader.batch_size

    class_counts: dict[int, int] = {}
    for _, labels in loader:
        for label in labels.tolist():
            class_counts[label] = class_counts.get(label, 0) + 1

    return {
        "n_samples":  n_samples,
        "n_batches":  n_batches,
        "batch_size": batch_size,
        "class_dist": dict(sorted(class_counts.items())),
    }
