import numpy as np
import pandas as pd

LANDMARK_CONFIG = {
    "pose": 33,
    "left_hand": 21,
    "right_hand": 21,
}

FEATURE_DIM = sum(
    count * 2 for count in LANDMARK_CONFIG.values()
)

def extract_landmarks(df):
    df = df[df["type"].isin(LANDMARK_CONFIG)].copy()
    df = df[["frame", "type", "landmark_index", "x", "y"]]

    frames = np.sort(df["frame"].unique())
    frame_to_idx = {
        frame: idx
        for idx, frame in enumerate(frames)
    }

    # Initialize missing coordinates with NaN.
    features = np.full(
        (len(frames), FEATURE_DIM),
        np.nan,
        dtype=np.float32
    )

    offset = 0

    for landmark_type, num_landmarks in LANDMARK_CONFIG.items():
        subset = df[df["type"] == landmark_type]

        frame_idx = subset["frame"].map(
            frame_to_idx
        ).to_numpy(dtype=np.int64)

        landmark_idx = subset[
            "landmark_index"
        ].to_numpy(dtype=np.int64)

        x = subset["x"].to_numpy(dtype=np.float32)
        y = subset["y"].to_numpy(dtype=np.float32)

        valid = (
            (landmark_idx >= 0) &
            (landmark_idx < num_landmarks)
        )

        frame_idx = frame_idx[valid]
        landmark_idx = landmark_idx[valid]

        values = np.column_stack([
            x[valid],
            y[valid]
        ])

        # Each landmark occupies two consecutive features.
        feature_idx = offset + landmark_idx * 2

        features[
            frame_idx[:, None],
            feature_idx[:, None] + np.array([0, 1])
        ] = values

        offset += num_landmarks * 2

    return features