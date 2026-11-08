from pathlib import Path
import pandas as pd
import numpy as np
import json

from processing import extract_landmarks
from normalization import normalize_sequence

# ============================================================
# PATH CONFIGURATION
# ============================================================

RAW_DIR = Path("data/raw")
LANDMARK_DIR = RAW_DIR / "train_landmark_files"

PROCESSED_DIR = Path("data/processed")

RAW_FEATURE_DIR = PROCESSED_DIR / "raw_features"
NORMALIZED_DIR = PROCESSED_DIR / "normalized"

METADATA_PATH = RAW_DIR / "metadata.csv"

MANIFEST_PATH = PROCESSED_DIR / "manifest.csv"
LABEL2ID_PATH = PROCESSED_DIR / "label2id.json"
ID2LABEL_PATH = PROCESSED_DIR / "id2label.json"

# ============================================================
# CONFIGURATION
# ============================================================

FEATURE_DIM = 150

# Skip processing if the corresponding output already exists.
SKIP_EXISTING = True

# ============================================================
# PATH HELPERS
# ============================================================

def get_feature_path(base_dir, participant_id, sequence_id):
    return (
        base_dir
        / str(participant_id)
        / f"{sequence_id}.npy"
    )

# ============================================================
# EXTRACTION
# ============================================================

def extract_and_save(parquet_path, output_path):
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df = pd.read_parquet(parquet_path)

    features = extract_landmarks(df)

    if len(features) == 0:
        raise ValueError(
            f"Empty sequence: {parquet_path}"
        )

    np.save(output_path, features)

    return features

# ============================================================
# NORMALIZATION
# ============================================================

def normalize_and_save(features, output_path):
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    normalized = normalize_sequence(features)

    normalized = np.nan_to_num(
        normalized,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    ).astype(np.float32)

    np.save(output_path, normalized)

    return normalized

# ============================================================
# VALIDATION
# ============================================================

def validate_features(features, expected_dim=FEATURE_DIM):
    if features.ndim != 2:
        raise ValueError(
            f"Expected 2D features, got {features.shape}"
        )

    if features.shape[1] != expected_dim:
        raise ValueError(
            f"Expected feature dimension {expected_dim}, "
            f"got {features.shape[1]}"
        )

# ============================================================
# DATASET PREPARATION
# ============================================================

def prepare_dataset():

    # --------------------------------------------------------
    # 1. Load metadata and label mappings
    # --------------------------------------------------------

    metadata = pd.read_csv(METADATA_PATH)

    selected_signs = sorted(metadata["sign"].unique())

    label2id = {
        sign: idx
        for idx, sign in enumerate(selected_signs)
    }

    # Create output directories.
    for directory in [
        PROCESSED_DIR,
        RAW_FEATURE_DIR,
        NORMALIZED_DIR
    ]:
        directory.mkdir(
            parents=True,
            exist_ok=True
        )

    manifest = []

    extraction_count = 0
    normalization_count = 0
    skipped_count = 0

    # --------------------------------------------------------
    # 2. Process each recording
    # --------------------------------------------------------

    for _, row in metadata.iterrows():

        participant_id = str(row["participant_id"])
        sequence_id = str(row["sequence_id"])
        sign = row["sign"]

        parquet_path = (
            LANDMARK_DIR
            / participant_id
            / f"{sequence_id}.parquet"
        )

        raw_path = get_feature_path(
            RAW_FEATURE_DIR,
            participant_id,
            sequence_id
        )

        normalized_path = get_feature_path(
            NORMALIZED_DIR,
            participant_id,
            sequence_id
        )

        # ----------------------------------------------------
        # 2A. Extract raw features
        # ----------------------------------------------------

        if raw_path.exists() and SKIP_EXISTING:
            features = np.load(raw_path)

        else:
            if not parquet_path.exists():
                print(
                    f"Missing Parquet file: {parquet_path}"
                )
                skipped_count += 1
                continue

            try:
                features = extract_and_save(
                    parquet_path,
                    raw_path
                )

                extraction_count += 1

            except Exception as e:
                print(
                    f"Extraction failed: {sequence_id} | {e}"
                )
                skipped_count += 1
                continue

        # Validate raw features.
        try:
            validate_features(features)

        except ValueError as e:
            print(
                f"Invalid raw features: {sequence_id} | {e}"
            )
            skipped_count += 1
            continue

        # ----------------------------------------------------
        # 2B. Normalize features
        # ----------------------------------------------------

        if normalized_path.exists() and SKIP_EXISTING:
            normalized = np.load(normalized_path)

        else:
            try:
                normalized = normalize_and_save(
                    features,
                    normalized_path
                )

                normalization_count += 1

            except Exception as e:
                print(
                    f"Normalization failed: {sequence_id} | {e}"
                )
                skipped_count += 1
                continue

        # Validate normalized features.
        try:
            validate_features(normalized)

            if not np.isfinite(normalized).all():
                raise ValueError(
                    "Normalized features contain NaN or Inf."
                )

        except ValueError as e:
            print(
                f"Invalid normalized features: {sequence_id} | {e}"
            )
            skipped_count += 1
            continue

        # ----------------------------------------------------
        # 2C. Add recording to manifest
        # ----------------------------------------------------

        manifest.append({
            "participant_id": participant_id,
            "sequence_id": sequence_id,
            "sign": sign,
            "label": label2id[sign],
            "n_frames": len(normalized),
            "path": str(normalized_path)
        })

    # --------------------------------------------------------
    # 3. Generate manifest
    # --------------------------------------------------------

    manifest_df = pd.DataFrame(manifest)

    manifest_df.to_csv(
        MANIFEST_PATH,
        index=False
    )

    # --------------------------------------------------------
    # 4. Save label mappings
    # --------------------------------------------------------

    with open(LABEL2ID_PATH, "w") as f:
        json.dump(
            label2id,
            f,
            indent=4
        )

    with open(ID2LABEL_PATH, "w") as f:
        json.dump(
            {v: k for k, v in label2id.items()},
            f,
            indent=4
        )

    # --------------------------------------------------------
    # 5. Report
    # --------------------------------------------------------

    print("\n========== PREPROCESSING COMPLETED ==========")

    print("Total metadata records:", len(metadata))
    print("Successfully processed:", len(manifest_df))
    print("Skipped recordings:", skipped_count)

    print("\nExtraction:")
    print("Newly extracted:", extraction_count)

    print("\nNormalization:")
    print("Newly normalized:", normalization_count)

    print("\nNumber of classes:", len(label2id))
    print("Feature dimension:", FEATURE_DIM)

    print("\nClass mapping:")
    print(label2id)

    print("\nSequence length statistics:")
    print(manifest_df["n_frames"].describe())

    print("\nManifest saved to:", MANIFEST_PATH)

# ============================================================
# ENTRY POINT
# ============================================================

def main():
    prepare_dataset()

if __name__ == "__main__":
    main()