"""
data/mnist.py

MNIST data loading for hyperparameter sweeps and fast iteration.

Entry points
------------
    get_mnist_loaders()   ->  (train_loader, val_loader, test_loader)
    get_probe_loader()    ->  fixed 512-sample loader for CKA / landscape viz

Two modes
---------
    Local dev  : pass subset_size=5000  -> 5k train, 1k val, fast iteration
    Full run   : pass subset_size=None  -> 60k train, 10k test, full MNIST

Augmentation
------------
    Train  : ToTensor + Normalize
    Val    : ToTensor + Normalize  (no augmentation — stable eval metrics)
    Probe  : ToTensor + Normalize  (fixed seed, never shuffled)

MNIST channel statistics (computed over full training set):
    mean = (0.1307,)
    std  = (0.3081,)

Note on input channels
----------------------
MNIST is grayscale (1 channel). TinyNet expects 3-channel input (CIFAR-10).
Pass in_channels=1 when constructing TinyNet for MNIST runs, or use the
repeat-channel transform below via the `expand_channels` flag.
"""

import torch
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms
from pathlib import Path


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MNIST_MEAN = (0.1307,)
MNIST_STD  = (0.3081,)

# Default download location — sits next to this file: data/mnist/
_DEFAULT_DATA_ROOT = Path(__file__).parent / "mnist"


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

def _base_transform(expand_channels: bool = False) -> transforms.Compose:
    """
    Standard MNIST transform: ToTensor + Normalize.

    Parameters
    ----------
    expand_channels : bool
        If True, repeat the single grayscale channel 3 times to produce a
        (3, 28, 28) tensor. Use this when feeding MNIST into a model that
        expects 3-channel input (e.g. the default TinyNet).
        If False, output is (1, 28, 28).
    """
    ops = [
        transforms.ToTensor(),
        transforms.Normalize(MNIST_MEAN, MNIST_STD),
    ]
    if expand_channels:
        ops.append(transforms.Lambda(lambda x: x.repeat(3, 1, 1)))
    return transforms.Compose(ops)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def get_mnist_loaders(
    data_root: str | Path = _DEFAULT_DATA_ROOT,
    batch_size: int = 128,
    subset_size: int | None = 5000,
    val_fraction: float = 0.1,
    num_workers: int = 0,
    seed: int = 42,
    expand_channels: bool = True,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Return (train_loader, val_loader, test_loader) for MNIST.

    The dataset is downloaded automatically on first call to `data_root`.

    Parameters
    ----------
    data_root : str or Path
        Directory where MNIST will be downloaded / read from.
        Defaults to data/mnist/ next to this file.
        On Colab, set to '/content/data' or '/content/drive/MyDrive/data'.

    batch_size : int
        Mini-batch size for all loaders. Default 128.

    subset_size : int or None
        If set, use only this many samples from the MNIST training set.
        The subset is drawn with a fixed seed for reproducibility.
        Pass None to use the full 60k training set.

        Recommended:
            Hyperparameter sweep  ->  subset_size=None   (fast enough on GPU)
            Local dev             ->  subset_size=5000

    val_fraction : float
        Fraction of the (possibly subsetted) training data to hold out
        for validation. Default 0.1 -> 10% val, 90% train.

    num_workers : int
        DataLoader worker processes. Default 0.

    seed : int
        RNG seed for the train/val split and subset selection.

    expand_channels : bool
        If True, repeat the grayscale channel 3 times to produce (3, 28, 28)
        tensors compatible with the default TinyNet architecture.
        Default True.

    Returns
    -------
    train_loader : DataLoader   (shuffled)
    val_loader   : DataLoader   (not shuffled)
    test_loader  : DataLoader   (not shuffled, official MNIST test set)
    """
    data_root = Path(data_root)
    tfm = _base_transform(expand_channels=expand_channels)

    # ── Download / load raw datasets ──────────────────────────────────────
    full_train = datasets.MNIST(data_root, train=True,  download=True, transform=tfm)
    test_set   = datasets.MNIST(data_root, train=False, download=True, transform=tfm)

    # ── Optional subsetting ───────────────────────────────────────────────
    rng = torch.Generator().manual_seed(seed)

    if subset_size is not None:
        n_total = len(full_train)
        if subset_size > n_total:
            raise ValueError(
                f"subset_size={subset_size} exceeds available training samples ({n_total})."
            )
        indices    = torch.randperm(n_total, generator=rng)[:subset_size].tolist()
        train_pool = Subset(full_train, indices)
    else:
        train_pool = full_train

    # ── Train / val split ─────────────────────────────────────────────────
    n_pool  = len(train_pool)
    n_val   = max(1, int(n_pool * val_fraction))
    n_train = n_pool - n_val

    train_subset, val_subset = random_split(
        train_pool, [n_train, n_val], generator=rng
    )

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
    expand_channels: bool = True,
) -> DataLoader:
    """
    Return a fixed, non-shuffled loader over a held-out probe set.

    Used for CKA representational similarity and loss landscape visualization.
    Drawn from the MNIST test set (never seen during training) with a fixed
    seed so Gram matrices are comparable across checkpoints and agents.

    Parameters
    ----------
    data_root : str or Path
        Same root used for get_mnist_loaders().
    n_samples : int
        Number of probe images. Default 512.
        Must be <= 10000 (MNIST test set size).
    batch_size : int
        Batch size for the probe loader. Default 512 (loads all at once).
    seed : int
        Fixed seed for probe index selection. Never change this between runs.
    expand_channels : bool
        Match the setting used in get_mnist_loaders(). Default True.

    Returns
    -------
    DataLoader (not shuffled, fixed order, same images every call)
    """
    data_root = Path(data_root)
    tfm = _base_transform(expand_channels=expand_channels)

    test_set = datasets.MNIST(data_root, train=False, download=True, transform=tfm)

    if n_samples > len(test_set):
        raise ValueError(
            f"n_samples={n_samples} exceeds test set size ({len(test_set)})."
        )

    rng       = torch.Generator().manual_seed(seed)
    indices   = torch.randperm(len(test_set), generator=rng)[:n_samples].tolist()
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
