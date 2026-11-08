import numpy as np
import pandas as pd
import torch

from pathlib import Path
from torch.utils.data import Dataset


class SignDataset(Dataset):

    def __init__(
        self,
        manifest_path="data/processed/split_manifest.csv",
        split="train"
    ):
        self.manifest_path = Path(manifest_path)
        self.root_dir = Path.cwd()

        if split not in {"train", "valid", "test"}:
            raise ValueError(
                f"Invalid split: {split}"
            )

        # Read manifest
        df = pd.read_csv(self.manifest_path)

        # Select the requested split
        self.df = df[df["split"] == split].reset_index(drop=True)

        if len(self.df) == 0:
            raise ValueError(
                f"No samples found for split: {split}"
            )

        print(
            f"{split.capitalize()} dataset: "
            f"{len(self.df)} samples"
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):

        row = self.df.iloc[index]

        # Resolve sequence path
        relative_path = Path(str(row["path"]).replace("\\", "/"))
        sequence_path = self.root_dir / relative_path

        if not sequence_path.exists():
            raise FileNotFoundError(
                f"Sequence not found: {sequence_path}"
            )

        # Load normalized sequence
        features = np.load(sequence_path)

        # Validate sequence
        if features.ndim != 2 or features.shape[1] != 150:
            raise ValueError(
                f"Invalid feature shape at {sequence_path}: "
                f"{features.shape}"
            )

        if len(features) != row["n_frames"]:
            raise ValueError(
                f"Frame count mismatch at {sequence_path}: "
                f"manifest={row['n_frames']}, "
                f"actual={len(features)}"
            )

        if not np.isfinite(features).all():
            raise ValueError(
                f"Non-finite values found in {sequence_path}"
            )

        # Convert to PyTorch tensors
        features = torch.from_numpy(
            features.astype(np.float32)
        )

        label = torch.tensor(
            row["label"],
            dtype=torch.long
        )

        return features, label