import torch
from torch.utils.data import DataLoader

from .dataset import SignDataset

def collate_fn(batch):
    if not batch:
        raise ValueError("Received an empty batch.")

    sequences, labels = zip(*batch)

    feature_dim = sequences[0].shape[1]

    if feature_dim != 150:
        raise ValueError(f"Expected 150 features, got {feature_dim}")

    lengths = torch.tensor(
        [len(seq) for seq in sequences],
        dtype=torch.long
    )

    if torch.any(lengths <= 0):
        raise ValueError("Sequences must contain at least one frame.")

    batch_size = len(sequences)
    max_length = lengths.max().item()

    padded = torch.zeros(
        batch_size,
        max_length,
        feature_dim,
        dtype=torch.float32
    )

    mask = torch.zeros(
        batch_size,
        max_length,
        dtype=torch.bool
    )

    for i, seq in enumerate(sequences):
        if seq.ndim != 2 or seq.shape[1] != feature_dim:
            raise ValueError(f"Invalid sequence shape: {seq.shape}")

        seq_len = len(seq)

        padded[i, :seq_len] = seq
        mask[i, :seq_len] = True

    labels = torch.stack(labels).long()

    return {
        "features": padded,
        "labels": labels,
        "lengths": lengths,
        "mask": mask
    }


def create_dataloader(
    dataset,
    batch_size=32,
    shuffle=False,
    num_workers=0
):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available()
    )

def get_loaders(
    manifest_path="data/processed/split_manifest.csv",
    batch_size=32,
    num_workers=0
):

    # --------------------------------------------------------
    # 1. Create datasets
    # --------------------------------------------------------

    train_dataset = SignDataset(
        manifest_path=manifest_path,
        split="train"
    )

    valid_dataset = SignDataset(
        manifest_path=manifest_path,
        split="valid"
    )

    test_dataset = SignDataset(
        manifest_path=manifest_path,
        split="test"
    )


    # --------------------------------------------------------
    # 2. Create DataLoaders
    # --------------------------------------------------------

    train_loader = create_dataloader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    valid_loader = create_dataloader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    test_loader = create_dataloader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    train_eval_loader = create_dataloader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return (
        train_loader,
        valid_loader,
        test_loader,
        train_eval_loader
    )