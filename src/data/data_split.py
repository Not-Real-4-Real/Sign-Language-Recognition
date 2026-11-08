from pathlib import Path
import numpy as np
import pandas as pd

MANIFEST_PATH = Path("data/processed/manifest.csv")
OUTPUT_PATH = Path("data/processed/split_manifest.csv")

SEED = 42
N_TRIALS = 10000

def create_split():
    df = pd.read_csv(MANIFEST_PATH)
    
    # 1. Build participant-class matrix
    participant_class = pd.crosstab(
        df["participant_id"],
        df["sign"]
    )

    participant_counts = participant_class.sum(axis=1)

    # Sort participants by recording count
    sorted_ids = participant_counts.sort_values().index.to_numpy()

    # Divide into three groups of seven
    groups = np.array_split(sorted_ids, 3)

    print("Participant groups:")
    for i, group in enumerate(groups):
        print(
            f"Group {i + 1}:",
            [(pid, int(participant_counts[pid])) for pid in group]
        )

    # 2. Define ideal class distributions
    ratios = {
        "train": 0.70,
        "valid": 0.15,
        "test": 0.15
    }

    global_class_counts = participant_class.sum(axis=0)

    target = {
        split: global_class_counts * ratio
        for split, ratio in ratios.items()
    }

    # 3. Randomized search
    rng = np.random.default_rng(SEED)

    best_score = float("inf")
    best_split = None

    for _ in range(N_TRIALS):

        valid_ids = []
        test_ids = []

        # Select one validation and one test participant from each recording-count group
        for group in groups:

            selected = rng.choice(
                group,
                size=2,
                replace=False
            )

            valid_ids.append(selected[0])
            test_ids.append(selected[1])

        valid_ids = set(valid_ids)
        test_ids = set(test_ids)

        train_ids = (
            set(sorted_ids) - valid_ids - test_ids
        )

        assignments = {
            "train": train_ids,
            "valid": valid_ids,
            "test": test_ids
        }

        # Evaluate class distribution balance
        score = 0.0

        for split, participant_ids in assignments.items():

            actual = participant_class.loc[
                list(participant_ids)
            ].sum(axis=0)

            expected = target[split]

            # Relative squared error
            relative_error = (
                (actual - expected) / expected
            ) ** 2

            score += relative_error.sum()

        if score < best_score:
            best_score = score
            best_split = assignments


    # 4. Assign split to every recording
    participant_to_split = {}

    for split, participant_ids in best_split.items():

        for pid in participant_ids:
            participant_to_split[pid] = split

    df["split"] = df["participant_id"].map(
        participant_to_split
    )

    # Save
    df.to_csv(OUTPUT_PATH, index=False)

    # 5. Print diagnostics
    print("\nSelected participants:")

    for split in ["train", "valid", "test"]:

        ids = sorted(best_split[split])

        print(f"\n{split.upper()} ({len(ids)} participants)")
        print(ids)

    print("\nSample distribution:")
    print(df["split"].value_counts())

    print("\nSample percentages:")
    print(
        df["split"].value_counts(normalize=True) * 100
    )

    print("\nClass distribution by split:")

    print(
        pd.crosstab(
            df["sign"],
            df["split"]
        )
    )

    print("\nParticipant counts by split:")
    print(
        df.groupby("split")["participant_id"].nunique()
    )

    print("\nSaved to:", OUTPUT_PATH)


if __name__ == "__main__":
    create_split()
