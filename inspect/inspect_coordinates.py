from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

RAW_DIR = Path("data/raw/train_landmark_files")
MANIFEST_PATH = Path("data/processed/split_manifest.csv")

SEED = 42
SAMPLES_PER_SPLIT = 100

LANDMARK_TYPES = ["pose", "left_hand", "right_hand"]

def load_sampled_recordings():

    manifest = pd.read_csv(MANIFEST_PATH)

    sampled = (
        manifest.groupby("split", group_keys=False)
        .sample(
            n=SAMPLES_PER_SPLIT,
            random_state=SEED
        )
        .reset_index(drop=True)
    )

    return sampled

def load_parquet(row):

    participant_id = str(row["participant_id"])
    sequence_id = str(row["sequence_id"])

    path = (
        RAW_DIR
        / participant_id
        / f"{sequence_id}.parquet"
    )

    return pd.read_parquet(path)

def inspect_coordinates():

    sampled = load_sampled_recordings()

    print("Sampled recordings:", len(sampled))
    print(sampled["split"].value_counts())

    # Collect coordinates by landmark component.
    coordinate_data = {
        landmark_type: {
            "x": [],
            "y": []
        }
        for landmark_type in LANDMARK_TYPES
    }

    missing_data = {
        landmark_type: {
            "total": 0,
            "missing_x": 0,
            "missing_y": 0,
            "missing_both": 0
        }
        for landmark_type in LANDMARK_TYPES
    }

    for _, row in sampled.iterrows():

        df = load_parquet(row)

        df = df[
            df["type"].isin(LANDMARK_TYPES)
        ]

        for landmark_type in LANDMARK_TYPES:

            subset = df[df["type"] == landmark_type]

            x = subset["x"].to_numpy(dtype=np.float32)
            y = subset["y"].to_numpy(dtype=np.float32)

            coordinate_data[landmark_type]["x"].append(x)
            coordinate_data[landmark_type]["y"].append(y)

            stats = missing_data[landmark_type]

            stats["total"] += len(subset)
            stats["missing_x"] += np.isnan(x).sum()
            stats["missing_y"] += np.isnan(y).sum()
            stats["missing_both"] += (
                np.isnan(x) & np.isnan(y)
            ).sum()

    # --------------------------------------------------
    # 1. Coordinate statistics
    # --------------------------------------------------

    print("\n=== COORDINATE STATISTICS ===")

    for landmark_type in LANDMARK_TYPES:

        print(f"\n{landmark_type.upper()}")

        for axis in ["x", "y"]:

            values = np.concatenate(
                coordinate_data[landmark_type][axis]
            )

            finite = values[np.isfinite(values)]

            print(f"\n{axis.upper()}")

            print("Total:", len(values))
            print("Finite:", len(finite))
            print("NaN:", np.isnan(values).sum())
            print("Inf:", np.isinf(values).sum())

            print(
                pd.Series(finite).describe(
                    percentiles=[
                        0.01,
                        0.05,
                        0.25,
                        0.50,
                        0.75,
                        0.95,
                        0.99
                    ]
                )
            )

            print("Outside [0, 1]:", (
                (finite < 0) | (finite > 1)
            ).sum())

    # --------------------------------------------------
    # 2. Missing coordinate statistics
    # --------------------------------------------------

    print("\n=== MISSING COORDINATES ===")

    for landmark_type, stats in missing_data.items():

        total = stats["total"]

        print(f"\n{landmark_type.upper()}")
        print("Total landmark observations:", total)

        if total > 0:
            print(
                "Missing x:",
                f'{stats["missing_x"] / total * 100:.2f}%'
            )

            print(
                "Missing y:",
                f'{stats["missing_y"] / total * 100:.2f}%'
            )

            print(
                "Both missing:",
                f'{stats["missing_both"] / total * 100:.2f}%'
            )

    # --------------------------------------------------
    # 3. Coordinate distributions
    # --------------------------------------------------

    fig, axes = plt.subplots(
        3, 2,
        figsize=(14, 12)
    )

    for i, landmark_type in enumerate(LANDMARK_TYPES):

        for j, axis in enumerate(["x", "y"]):

            values = np.concatenate(
                coordinate_data[landmark_type][axis]
            )

            values = values[np.isfinite(values)]

            axes[i, j].hist(
                values,
                bins=100
            )

            axes[i, j].set_title(
                f"{landmark_type} - {axis}"
            )

            axes[i, j].set_xlabel("Coordinate value")
            axes[i, j].set_ylabel("Frequency")

    plt.tight_layout()
    plt.show()

def inspect_pose_landmarks():

    manifest = pd.read_csv(MANIFEST_PATH)

    # Select a few recordings.
    samples = (
        manifest.groupby("split", group_keys=False)
        .sample(n=3, random_state=42)
    )

    for _, row in samples.iterrows():

        df = load_parquet(row)

        pose = df[df["type"] == "pose"].copy()

        print("\n" + "=" * 60)
        print(
            f"Participant: {row['participant_id']} | "
            f"Sequence: {row['sequence_id']} | "
            f"Sign: {row['sign']} | "
            f"Split: {row['split']}"
        )

        # Examine individual landmarks across frames.
        selected_indices = [0, 11, 12, 13, 14, 15, 16, 23, 24]

        for idx in selected_indices:

            landmark = pose[
                pose["landmark_index"] == idx
            ]

            print(
                f"Landmark {idx:2d}: "
                f"x=[{landmark['x'].min():.3f}, "
                f"{landmark['x'].max():.3f}], "
                f"y=[{landmark['y'].min():.3f}, "
                f"{landmark['y'].max():.3f}]"
            )

        print("\nFirst 5 frames:")
        print(
            pose[
                pose["landmark_index"].isin(selected_indices)
            ][
                ["frame", "landmark_index", "x", "y"]
            ].head(45).to_string(index=False)
        )

if __name__ == "__main__":
    # inspect_coordinates()
    inspect_pose_landmarks()
