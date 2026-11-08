from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from inspect_coordinates import load_parquet, load_sampled_recordings

RAW_DIR = Path("data/raw/train_landmark_files")
MANIFEST_PATH = Path("data/processed/split_manifest.csv")


def extract_landmark_sequence(df, landmark_index):
    subset = df[
        (df["type"] == "pose") &
        (df["landmark_index"] == landmark_index)
    ].sort_values("frame")

    return subset[["frame", "x", "y"]].to_numpy()

def inspect_recording(df, title):
    landmarks = {
        "nose": 0,
        "left_shoulder": 11,
        "right_shoulder": 12,
        "left_elbow": 13,
        "right_elbow": 14,
        "left_wrist": 15,
        "right_wrist": 16,
        "left_hip": 23,
        "right_hip": 24,
    }

    sequences = {
        name: extract_landmark_sequence(df, idx)
        for name, idx in landmarks.items()
    }

    left_shoulder = sequences["left_shoulder"]
    right_shoulder = sequences["right_shoulder"]

    # Match frames explicitly.
    left = pd.DataFrame(
        left_shoulder,
        columns=["frame", "lx", "ly"]
    )

    right = pd.DataFrame(
        right_shoulder,
        columns=["frame", "rx", "ry"]
    )

    shoulders = left.merge(right, on="frame")

    center_x = (shoulders["lx"] + shoulders["rx"]) / 2
    center_y = (shoulders["ly"] + shoulders["ry"]) / 2

    shoulder_width = np.sqrt(
        (shoulders["lx"] - shoulders["rx"]) ** 2 +
        (shoulders["ly"] - shoulders["ry"]) ** 2
    )

    # --------------------------------------------------
    # 1. Print geometric statistics
    # --------------------------------------------------

    print("\n", "=" * 60)
    print(title)

    print("\nShoulder midpoint:")
    print("X:", center_x.describe())
    print("Y:", center_y.describe())

    print("\nShoulder width:")
    print(shoulder_width.describe())

    # --------------------------------------------------
    # 2. Plot landmark trajectories
    # --------------------------------------------------

    fig, axes = plt.subplots(
        3, 1,
        figsize=(13, 10),
        sharex=True
    )

    axes[0].plot(
        shoulders["frame"],
        center_x,
        label="Shoulder center X"
    )

    axes[0].set_ylabel("X")
    axes[0].legend()

    axes[1].plot(
        shoulders["frame"],
        center_y,
        label="Shoulder center Y"
    )

    axes[1].set_ylabel("Y")
    axes[1].legend()

    axes[2].plot(
        shoulders["frame"],
        shoulder_width,
        label="Shoulder width"
    )

    axes[2].set_ylabel("Distance")
    axes[2].set_xlabel("Frame")
    axes[2].legend()

    fig.suptitle(title)
    plt.tight_layout()
    plt.show()

    # --------------------------------------------------
    # 3. Inspect wrist displacement
    # --------------------------------------------------

    fig, axes = plt.subplots(
        2, 1,
        figsize=(13, 7),
        sharex=True
    )

    for name in ["left_wrist", "right_wrist"]:

        seq = sequences[name]

        coords = seq[:, 1:3]

        displacement = np.linalg.norm(
            np.diff(coords, axis=0),
            axis=1
        )

        frames = seq[1:, 0]

        axes[0].plot(
            frames,
            coords[1:, 0],
            label=f"{name} X"
        )

        axes[1].plot(
            frames,
            displacement,
            label=f"{name} displacement"
        )

    axes[0].set_ylabel("X coordinate")
    axes[1].set_ylabel("Frame displacement")
    axes[1].set_xlabel("Frame")

    axes[0].legend()
    axes[1].legend()

    plt.tight_layout()
    plt.show()

def inspect_dataset_geometry():
    manifest = pd.read_csv(MANIFEST_PATH)

    results = []

    for _, row in manifest.iterrows():

        df = load_parquet(row)

        pose = df[df["type"] == "pose"]

        shoulders = pose[
            pose["landmark_index"].isin([11, 12])
        ]

        pivot = shoulders.pivot(
            index="frame",
            columns="landmark_index",
            values=["x", "y"]
        )

        # Require both shoulders to be available.
        pivot = pivot.dropna()

        if len(pivot) == 0:
            continue

        lx = pivot[("x", 11)].to_numpy()
        ly = pivot[("y", 11)].to_numpy()

        rx = pivot[("x", 12)].to_numpy()
        ry = pivot[("y", 12)].to_numpy()

        center_x = (lx + rx) / 2
        center_y = (ly + ry) / 2

        width = np.sqrt(
            (lx - rx) ** 2 +
            (ly - ry) ** 2
        )

        results.append({
            "participant_id": row["participant_id"],
            "sequence_id": row["sequence_id"],
            "sign": row["sign"],
            "split": row["split"],

            "center_x_mean": center_x.mean(),
            "center_y_mean": center_y.mean(),

            "center_x_std": center_x.std(),
            "center_y_std": center_y.std(),

            "width_mean": width.mean(),
            "width_std": width.std(),

            "width_cv": (
                width.std() / width.mean()
                if width.mean() > 0 else np.nan
            )
        })

    geometry = pd.DataFrame(results)

    print("\n=== DATASET GEOMETRY ===")

    print("\nShoulder center X:")
    print(geometry["center_x_mean"].describe())

    print("\nShoulder center Y:")
    print(geometry["center_y_mean"].describe())

    print("\nMean shoulder width:")
    print(geometry["width_mean"].describe())

    print("\nShoulder width CV:")
    print(geometry["width_cv"].describe())

    print("\nWithin-recording shoulder center variation:")
    print(
        geometry[
            ["center_x_std", "center_y_std"]
        ].describe()
    )

    print("\nInvalid or unusually small shoulder widths:")
    print(
        geometry[
            (geometry["width_mean"] < 0.05) |
            (geometry["width_mean"].isna())
        ].shape[0]
    )

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    axes[0, 0].hist(geometry["center_x_mean"], bins=50)
    axes[0, 0].set_title("Recording-level shoulder center X")

    axes[0, 1].hist(geometry["center_y_mean"], bins=50)
    axes[0, 1].set_title("Recording-level shoulder center Y")

    axes[1, 0].hist(geometry["width_mean"], bins=50)
    axes[1, 0].set_title("Mean shoulder width")

    axes[1, 1].hist(geometry["width_cv"], bins=50)
    axes[1, 1].set_title("Shoulder width CV")

    plt.tight_layout()
    plt.show()

    return geometry

def main():
    geometry = inspect_dataset_geometry()

    # samples = load_sampled_recordings()

    # # Inspect one recording from each split.
    # selected = (
    #     samples.groupby("split")
    #     .head(1)
    # )

    # for _, row in selected.iterrows():

    #     df = load_parquet(row)

    #     title = (
    #         f"Participant {row['participant_id']} | "
    #         f"Sign {row['sign']} | "
    #         f"Split {row['split']}"
    #     )

    #     inspect_recording(df, title)

if __name__ == "__main__":
    main()
